# 02 — Microarchitecture & RTL Reference

> Companion to `docs/01_system_architecture.md`. This document is the implementation reference:
> the module hierarchy, every port, the internal state machines, a **measured** cycle-by-cycle
> timeline, and the synthesis rules the RTL follows. All cycle numbers below were captured from an
> Icarus Verilog simulation of `llm_decode_top` at the default parameters (N = 4, d_k = 16).

---

## 1. Module Hierarchy

```
llm_decode_top                      rtl/llm_decode_top.v        AXI4-Stream wrapper, parser, engine
├── u_kv      kv_cache_controller   rtl/kv_cache_controller.v   sink/window KV SRAMs, INT4 quantise+pack
├── u_unpack  kv_unpack_dequant     rtl/kv_unpack_dequant.v     INT4x2 -> INT8 lanes (1 pipeline stage)
├── u_pp      ping_pong_buffer      rtl/ping_pong_buffer.v      dual-bank Q^T / P^T operand buffer
└── u_array   systolic_array        rtl/systolic_array.v        N x N mesh + skew registers
    └── g_r[i].g_c[j].u_pe  pe      rtl/pe.v                    INT8 MAC, 2-stage pipeline
```

Every module is plain Verilog-2001 (`generate`, `localparam`, constant functions, and no
SystemVerilog). It elaborates cleanly in Icarus Verilog (`-g2001`) and Yosys with no latches and no
multiple drivers.

---

## 2. Signal Dictionaries

### 2.1 `pe` — Processing Element

| Port        | Dir | Width  | Description                                                     |
|-------------|-----|--------|-----------------------------------------------------------------|
| `clk`       | in  | 1      | Clock                                                           |
| `rst_n`     | in  | 1      | Synchronous active-low reset (control registers only)           |
| `a_in`      | in  | 8 (s)  | Row operand from the west neighbour / skew register             |
| `b_in`      | in  | 8 (s)  | Column operand from the north neighbour / skew register         |
| `v_in`      | in  | 1      | Beat valid (travels with `a`)                                   |
| `f_in`      | in  | 1      | First beat of a reduction: the accumulator restarts from this product |
| `l_in`      | in  | 1      | Last beat of a reduction: the result is latched into `res`       |
| `a_out`     | out | 8 (s)  | `a_in` delayed one cycle → east neighbour                       |
| `b_out`     | out | 8 (s)  | `b_in` delayed one cycle → south neighbour                      |
| `v_out/f_out/l_out` | out | 1 | Control tags delayed one cycle → east neighbour           |
| `res`       | out | 32 (s) | Final dot product of the last completed reduction               |
| `res_valid` | out | 1      | One-cycle pulse when `res` updates                              |

Pipeline: **S0** operand registers → **S1** `prod ← a·b` (16-bit signed) → **S2**
`acc ← first ? prod : acc + prod`, with `res ← acc_next` on `last`. The product is sign-extended to
32 bits before accumulation. S0 lets the placer put the multiplier's input flops next to the DSP
block (docs/04, §3).

### 2.2 `systolic_array` — N × N mesh

| Port        | Dir | Width      | Description                                                     |
|-------------|-----|------------|-----------------------------------------------------------------|
| `a_in`      | in  | N·8        | Lane *i* = A[i][k] for the current reduction step k            |
| `b_in`      | in  | N·8        | Lane *j* = B[k][j]                                              |
| `in_valid`  | in  | 1          | Beat valid. When low, operands are **zero-padded** at the edge  |
| `in_first`  | in  | 1          | k = 0                                                           |
| `in_last`   | in  | 1          | k = K − 1                                                       |
| `c_flat`    | out | N·N·32     | C[i][j] at bits `[(i·N+j)·32 +: 32]`                           |
| `out_valid` | out | 1          | Pulse when PE(N−1,N−1) latches, meaning the whole tile is complete |

Skew: every lane first passes one **ingress register**, which isolates the operand SRAM and mux
path from the PE multipliers. Row *i* (operand `a` **and** its control tags) then passes through *i*
more registers, and column *j* through *j* more. A[i][k] and B[k][j] therefore meet in PE(i,j) at
cycle k + 1 + i + j. **Latency `in_last` → `out_valid` = 2(N−1) + 4**, which is 10 cycles for
N = 4 and 18 for N = 8.

### 2.3 `ping_pong_buffer`

| Port          | Dir | Width   | Description                                                  |
|---------------|-----|---------|--------------------------------------------------------------|
| `wr_en`       | in  | 1       | Write one word into the current fill bank (`wr_bank`)        |
| `wr_addr`     | in  | AW      | Word address                                                 |
| `wr_data`     | in  | N·8     | Word: lane *j* = operand for array column/row *j*           |
| `wr_commit`   | in  | 1       | Fill complete: bank → READY, `wr_bank` toggles               |
| `wr_ready`    | out | 1       | The fill bank is EMPTY or FILLING                            |
| `rd_acquire`  | in  | 1       | Consumer claims the READY bank: READY → COMPUTE              |
| `rd_release`  | in  | 1       | Consumer done: COMPUTE → EMPTY, `rd_bank` toggles            |
| `rd_en`/`rd_addr` | in | 1/AW | Synchronous read of the compute bank                         |
| `rd_data`     | out | N·8     | Read data, valid 1 (`OUT_REG=0`) or 2 (`OUT_REG=1`, default) cycles after `rd_en`, held otherwise |
| `rd_ready`    | out | 1       | The compute bank is READY                                    |
| `wr_bank`, `rd_bank` | out | 1 | Bank pointers                                            |
| `bank_state`  | out | 4       | `{state1, state0}`: 0 EMPTY, 1 FILLING, 2 READY, 3 COMPUTE   |

### 2.4 `kv_unpack_dequant`

| Port      | Dir | Width   | Description                                                     |
|-----------|-----|---------|-----------------------------------------------------------------|
| `shift`   | in  | 3       | Dequantisation exponent σ (values > 4 clamp to 4)              |
| `s_valid/s_ready/s_last` | in/out/in | 1 | Packed-row stream handshake                        |
| `s_data`  | in  | d_k·4   | Packed row: element e at `[4e+3:4e]`                            |
| `m_valid/m_ready/m_last` | out/in/out | 1 | Unpacked-row stream handshake                     |
| `m_data`  | out | d_k·8   | Lane e = `sext4→8(nibble_e) << σ`                               |

Parameter `PIPELINE` = 0 gives a purely combinational path. `PIPELINE` = 1 inserts one output
register stage with `s_ready = !m_valid || m_ready`, which runs at full rate with no bubbles.
`PIPELINE` = 2 (used in the top level) also registers the packed input, so the KV SRAM's
clock-to-out never meets the unpack logic in the same cycle.

### 2.5 `kv_cache_controller`

| Port          | Dir | Width   | Description                                                   |
|---------------|-----|---------|---------------------------------------------------------------|
| `clear`       | in  | 1       | Empty the cache (pointers and counters reset; data left stale) |
| `cfg_shift`   | in  | 3       | Quantisation exponent σ used on append                       |
| `app_valid/app_ready` | in/out | 1 | Append handshake. `app_ready` = 0 while a burst is active |
| `app_k/app_v` | in  | d_k·8   | INT8 key / value row of the new token                        |
| `rd_start`    | in  | 1       | Launch a two-phase burst                                     |
| `rd_sel`      | in  | 1       | 0 = stream K rows, 1 = stream V rows                         |
| `rd_busy`     | out | 1       | Burst in progress or output register still full              |
| `m_valid/m_ready` | out/in | 1 | Packed-row stream handshake                                 |
| `m_data`      | out | d_k·4   | Packed INT4 row                                              |
| `m_last`      | out | 1       | Final row of the burst                                       |
| `m_phase`     | out | 1       | 0 = row from Phase A (sink), 1 = Phase B (window)            |
| `token_count` | out | 32      | Tokens appended since reset/clear (saturating)              |
| `sink_fill`   | out | 16      | Sink slots in use (0…SINK_COUNT)                             |
| `win_fill`    | out | 16      | Window slots in use (0…WINDOW_SIZE)                          |
| `head_ptr`    | out | 32      | Free-running window write pointer (slot = `head_ptr & (W−1)`) |

Internal read FSM: `RD_IDLE → RD_SINK → RD_WIN → RD_IDLE`. `RD_SINK` is skipped if
`sink_fill = 0` and `RD_WIN` is skipped if `win_fill = 0`. The phase bounds (`sink_last`,
`win_last`) and the oldest window slot (`win_start`) are registered at burst start, so phase-end
detection is a narrow equality compare. The window address is a `log2(W)`-bit add that wraps for
free. The append side is a 4-stage pipeline: accept (pointers advance) → quantise+pack K → quantise+pack V
→ K/V SRAM write. A single d_k-lane quantiser is **time-shared** between K and V (halving its LUT
cost, see docs/04), so the controller accepts at most one append every 2 cycles. An APPEND already
spans 3 AXI beats, so this costs nothing. A `rd_start` that arrives while a write is in flight is deferred (`start_pending`) until
the write lands. The SRAM output register *is* the
stream register. The read enable is `rd_en = active && (!m_valid || m_ready)`, so the BRAM holds its
output under backpressure and no skid buffer is needed.

### 2.6 `llm_decode_top`

| Port             | Dir | Width  | Description                                                |
|------------------|-----|--------|------------------------------------------------------------|
| `s_axis_tdata`   | in  | d_k·8  | Command header / K,V row / buffer word (low N·8 bits)      |
| `s_axis_tvalid`  | in  | 1      | AXI4-Stream valid                                          |
| `s_axis_tready`  | out | 1      | AXI4-Stream ready (see the parser table)                   |
| `s_axis_tlast`   | in  | 1      | Ends a LOAD payload (required); informational otherwise    |
| `m_axis_tdata`   | out | OUT_LANES·32 | `OUT_LANES` signed INT32 results, lane k at `[32k+31:32k]` (default one array row) |
| `m_axis_tvalid`  | out | 1      | Result valid                                               |
| `m_axis_tready`  | in  | 1      | Downstream backpressure                                    |
| `m_axis_tlast`   | out | 1      | Last beat of a SCORE / CONTEXT result                      |
| `busy`           | out | 1      | Engine active or a packet is part-way through parsing      |
| `token_count`    | out | 32     | Mirrors the KV controller                                  |

Parameters: `N`, `HEAD_DIM` (a multiple of N, even), `SINK_COUNT`, `WINDOW_SIZE` (2^k, ≥ 2),
`BUF_DEPTH` (≥ max(HEAD_DIM, SINK_COUNT + WINDOW_SIZE)), `BUF_AW = clog2(BUF_DEPTH)`, and
`PP_OUT_REG` (ping-pong output register, default 1), and `OUT_LANES` (results per m_axis beat,
default N; must divide N).

Key internal signals, useful as waveform probes (they are also used in `make wave`):

| Signal          | Meaning                                                              |
|-----------------|----------------------------------------------------------------------|
| `p_state`       | Parser: 0 HDR, 1 APP_K, 2 APP_V, 3 LOAD, 4 WAIT                       |
| `e_state`       | Engine: 0 IDLE, 1 WBUF (wait for a bank), 2 RUN, 3 NEXT (next CONTEXT pass), 4 DRAIN |
| `tile_full`, `tile_rows`, `sc_d` | SCORE tile former / column index d                   |
| `cx_t`, `e_pass`| CONTEXT token index / pass index                                     |
| `feed_go`       | Beat issued to the array this cycle (bank read + operand register)  |
| `f_valid/f_first/f_last` | Registered feed tags (bank read issued)                     |
| `a_valid/a_first/a_last` | Array-input tags, aligned with the 2-cycle bank read        |
| `arr_done`      | Array tile complete                                                  |
| `pending`, `gap_cnt`, `gap_ok`, `hold` | Tile-overlap bookkeeping (§4.3)                |
| `cap_full_q`, `o_i`, `o_j` | Capture buffer and serializer indices                      |

---

## 3. Parser and Engine State Machines

### 3.1 Command parser (`p_state`)

```
            ┌────────────────────── s_hs & op=APPEND ──────► P_APP_K ──s_hs──► P_APP_V ─┐
            │                                                 (latch K)   (write K,V)  │
  ┌──────►P_HDR ──────────────── s_hs & op=LOAD ─────────► P_LOAD ──s_hs & tlast─────────┤
  │         │                                             (write word, addr++)         │
  │         └──── s_hs & op∈{CFG,SCORE,CONTEXT,CLEAR} ──► P_WAIT ──eng_idle──────────────┤
  │                                                        (apply / dispatch)          │
  └────────────────────────────────────────────────────────────────────────────────────┘
```

| State    | `s_axis_tready`                    | Rationale                                         |
|----------|------------------------------------|---------------------------------------------------|
| P_HDR    | 1                                  | Headers are always accepted                       |
| P_APP_K  | 1                                  | K row is only registered                          |
| P_APP_V  | `eng_idle & !eng_start & app_ready`| The KV write must not collide with a compute burst |
| P_LOAD   | `pp_wr_ready`                      | Stall when no bank is free (ping-pong backpressure) |
| P_WAIT   | 0                                  | Wait for the engine before CFG/CLEAR/dispatch     |

SCORE/CONTEXT leave P_WAIT in the same cycle they are dispatched. The parser is back in P_HDR while
the engine computes, and that is what allows a LOAD to fill the other bank concurrently.

> **Host ordering rule.** The stream is in-order. With both banks READY, a third `LOAD` would sit
> in P_LOAD holding the bus, and the SCORE/CONTEXT that would free a bank could never arrive. The
> host must therefore keep **at most two LOADs ahead of the compute commands that consume them**.
> `test_pingpong_prefetch` exercises the legal limit: two banks pre-loaded, and a third LOAD that
> stalls until compute releases a bank.

### 3.2 Compute engine (`e_state`)

```
 E_IDLE ──eng_start & token_count≠0──► E_WBUF ──pp_rd_ready──► E_RUN
    ▲        (acquire bank, rd_start)                         │  SCORE: tiles of N rows × d_k beats
    │                                                         │  CONTEXT: d_k/N passes × S beats
    │                                   E_NEXT ◄──pass done───┤  (restart the V burst for the next pass)
    │                                      └──kv idle──► E_RUN │
    └──── pending=0 & capture empty ──── E_DRAIN ◄──last beat fed (release bank)
```

The ping-pong bank is **released as soon as the final beat has been fed**, not when the results
have drained. The next LOAD can therefore start filling that bank while the output serializer is
still emptying.

---

## 4. Datapath Details

### 4.1 SCORE mapping

* The KV controller streams K rows (sinks, then the window oldest-first). The unpacker converts
  each row to d_k INT8 lanes.
* **Tile former:** N consecutive rows are latched into `tile_row[0..N-1]`. The final tile may be
  partial (`tile_rows < N`), and its unused rows are fed as zeros.
* **Feed:** for d = 0…d_k−1, `a` lane i = `tile_row[i][d]` (token i) and `b` lane j = bank word d
  lane j = Q[j][d]. The bank read and the operand register are issued in the same cycle, so both
  arrive at the array together one cycle later.
* Output: `tile_rows × N` words per tile, token-major.

### 4.2 CONTEXT mapping

* For pass p = 0…d_k/N−1, the V rows are streamed straight from the unpacker (no tile former).
  `b` lane j = V[t][pN+j] and `a` lane i = bank word t lane i = P[i][t].
* Every PE reduces over the whole context (K = S ≤ 68). Output per pass: N × N words, head-major.

### 4.3 Result capture and tile overlap

The array accumulates in place, and each PE keeps its result only until it sees the *next* tile's
`last`. The top level therefore double-buffers results with a **capture register** (N·N·32 bits,
loaded in one cycle on `arr_done`) plus three small counters:

| Rule | Condition | Why |
|------|-----------|-----|
| First beat of a tile may issue | `pending = 0` **or** (`pending = 1` and the capture buffer is empty) | At most one uncaptured tile can be in the mesh, and it is guaranteed a free capture slot |
| Last beat of a tile may issue  | `pending = 0` **or** `gap_ok` (≥ 2N+2 cycles since the previous last beat) | PE(0,0) latches the new tile 4 cycles after its last beat, while the previous tile completes 2(N−1)+4 cycles after its own. A spacing ≥ 2N−1 is sufficient, and 2N+2 adds margin |
| Late completion                | `arr_done` while the capture buffer is full → `hold` | Captured as soon as the serializer frees the buffer; no new tile starts meanwhile |

Consecutive SCORE tiles therefore overlap: tile t+1 computes while tile t drains, as the trace
below shows. The rules stay correct for CONTEXT passes as short as one token (K = 1). That case is
verified in `test_empty_and_reconfig`.

---

## 5. Measured Cycle-by-Cycle Timeline

Scenario: six tokens are appended (token *t* has all K/V elements = t+1), a Q^T LOAD
(16 words) follows, then `SCORE` with an always-ready master. The context is 6 tokens (4 sinks +
2 window), which gives **tile 0** = 4 rows (S0–S3) and **tile 1** = 2 rows (W0, W1, zero-padded to
4). The cycle numbers come from the Icarus simulation of the final RTL
(`PIPELINE=2` unpacker, `PP_OUT_REG=1`, 3-stage PEs).

### 5.1 Token ingestion → INT4 packing (per APPEND)

```
cycle            0        1        2        3        4        5
                 ┌────────┬────────┬────────┐
s_axis beat      │ HDR 0x1│ K row  │ V row  │  (next packet ...)
                 └────────┴────────┴────────┘
p_state          HDR      APP_K    APP_V    HDR
app_valid&ready                    ▲ pointers advance (sink_fill / head_ptr, token_count)
append pipe                                 W1: quantise+pack K (shared quantiser)
                                                     W2: quantise+pack V (same quantiser)
                                                              W3: K/V SRAM write @ slot
```

### 5.2 SCORE: sink read → window read → unpack → MAC → result

```
cycle        35   38 39 40 41 42 43 44 45 46 47 48 49 ··· 62 63 64 65 66 67 ··· 74 75 ··· 80 81 82 ··· 90 91 92 93 ··· 100
s_axis       SCR
e_state           WB RUN────────────────────────────────────────────────────────────────── DRAIN──────────────────────── IDLE→
kv rd FSM            ST SNK─────────WIN──IDLE
packed row             S0 S1 S2 S3 W0 W1                                       (Phase A = sinks, Phase B = window)
unpacked row                 S0 S1 S2 S3                   W0 W1               (W0/W1 wait in the 2-stage unpacker)
tile_full                                1 ────────────── 1  0  0  1 ──────────────────── 1  0
feed (d)                                 d0 d1 d2 ··· d15       d0 d1 d2 ··· d9 ··· d14 d15
array input                                    d0 ··· d13 d14 d15       d0 d1 ··· d7 d8 ··· d13 d14 d15
                                               ▲first         ▲last          ▲first               ▲last(82)
arr_done                                                                  ▲74 = 64 + 2(N-1)+4            ▲92 = 82 + 10
m_axis (tile 0)                                                              75 ─ 4 rows ─ 78
m_axis (tile 1)                                                                                93 ─ 94 tlast   (2 rows)
```

Observations from the trace:

* **Two-phase burst.** Rows leave the SRAM as `S0 S1 S2 S3` (Phase A, `m_phase = 0`) and then
  `W0 W1` (Phase B, `m_phase = 1`) on consecutive cycles. The FSM moves SINK → WIN with no bubble.
* **Backpressure absorption.** The window rows are read immediately but wait inside the
  unpacker's two register stages while tile 0 is fed. No SRAM re-read is needed.
* **Tile overlap.** Tile 1 is fed (cycles 65–80) while tile 0 is still draining through the mesh and
  the output port (75–78). With `OUT_LANES = N` one array row leaves per beat, so a 4×4 tile drains
  in 4 cycles (it took 16 with a 32-bit port).
* **Bank release.** The engine enters DRAIN at cycle 81, the cycle after the last feed beat. The
  ping-pong bank is released at that point, before the results have drained.

### 5.3 Per-stage latency summary (N = 4, d_k = 16)

| Stage                                   | Cycles        | Notes                                                |
|-----------------------------------------|---------------|------------------------------------------------------|
| APPEND header → K/V SRAM write          | 2 + 3         | 3 beats; K quantise (W1), V quantise (W2), write (W3) |
| SCORE header → engine RUN               | 4             | P_WAIT → eng_start → WBUF → acquire                  |
| `rd_start` → first packed row           | 2             | FSM entry + synchronous SRAM read                    |
| Packed row → unpacked row               | 2             | Unpacker input register + output register            |
| N rows → tile full                      | N             | One row per cycle                                    |
| Tile issue (feed)                       | d_k           | One reduction step per cycle                         |
| Feed beat → array input                 | 2             | Ping-pong read latency (BRAM + output register)      |
| Array input `last` → `arr_done`         | 2(N−1) + 4    | Ingress reg + skew/mesh traversal + 3 PE stages      |
| `arr_done` → first m_axis beat          | 1             | Capture register                                     |
| Tile drain through m_axis               | rows × N / OUT_LANES | 4 beats for a full 4×4 tile at the default width |
| **SCORE header → first result word**    | **40**        | Measured                                             |
| Steady state per full SCORE tile        | d_k + 2 ≈ 18  | Tile former is single-buffered (see docs/04)         |

## 6. Synthesis Guidelines Followed by the RTL

### 6.1 Latch prevention
* Every combinational `always @(*)` (only `sc_col` in the top level) assigns its output
  unconditionally in a `for` loop with a ternary default. There are no incomplete `if`/`case`
  statements in combinational logic.
* FSM `case` statements always carry a `default` that returns to a safe state.
* Functions (`quant4`, `next_state`, `clog2`) assign their return value on every path.
* Checked with Yosys `proc; check -assert`: no latches and no multiple drivers.

### 6.2 Reset domain strategy
* A single clock domain with a **synchronous, active-low `rst_n`**. That suits FPGA
  (it maps to the flop's SR/CE pins and does not use the global async net) and is timing-friendly
  in ASIC flows.
* **Only control state is reset**: FSMs, valid/first/last tags, pointers, counters, and bank
  states. Wide datapath registers (operands, products, accumulators, SRAM outputs, the capture
  buffer) are *reset-free*. They are always qualified by a reset valid bit, which removes a large
  reset fan-out and allows DSP/BRAM register absorption.
* `clear` is a synchronous soft reset of the KV pointers only, so the SRAM contents need no
  initialisation.
* If the reset source is asynchronous, the integrator must provide a reset synchroniser upstream
  (standard 2-flop assert-async / deassert-sync).

### 6.3 Memory inference and collision prevention
* Every SRAM (K, V, ping-pong bank 0/1) is written as a **simple dual-port** template: one write
  port and one registered read port with a read enable. That is the canonical pattern for Xilinx
  RAMB18/36, Lattice DP16KD, and ASIC 1R1W macros.
* **KV SRAMs:** `app_ready` is forced low while the read FSM is active, and the parser also holds
  APPEND until the engine is idle. A read and a write to the same address in the same cycle are
  therefore impossible, and no read-during-write mode needs to be specified. This is checked
  every cycle by `test_kv_cache` (the "collision guard" monitor).
* **Ping-pong banks:** a bank is writable only in EMPTY/FILLING and readable only in
  READY/COMPUTE. The two ports can never address the same bank.
* **Sinks:** window addresses are `SINK_COUNT + (head_ptr & (W−1)) ≥ SINK_COUNT`, so the window
  path cannot reach a sink slot.

### 6.4 Timing-oriented structure
* The PE has a three-stage pipeline (operand, multiply, accumulate). It maps onto DSP48E1
  `AREG/BREG`/`MREG`/`PREG`, or onto an ECP5 MULT18 plus an ALU carry chain.
* All mesh interconnect is nearest-neighbour and registered, so there are no global broadcast nets
  inside the array.
* The KV read path is SRAM → K/V select → register → unpack (sign-extend + ≤4-bit shift mux,
  1–2 LUT levels) → register. It is decoupled from the MAC path by the feed and ingress registers.
* The ping-pong read is SRAM → per-bank output register → bank/mode/zero-pad mux → array ingress
  register.
* The quantiser uses no adders or comparators wider than 4 bits. Round-half-up is computed as
  `(x >>> s) + x[s−1]`, and overflow is detected as "bits [7:3] are not all equal to the sign bit".
  The shared quantiser is fed from a dedicated input register (`q_x`/`q_s`) that loads K on
  accept and V one cycle later. The K/V selection therefore sits in front of a flop instead of
  the quantiser.
* The widest combinational fan-in is the capture-buffer serializer mux (N²/OUT_LANES : 1 over
  OUT_LANES·32 bits; N : 1 row select at the default width) and
  the tile column mux (d_k : 1 per lane). Both are analysed in `docs/04`.
