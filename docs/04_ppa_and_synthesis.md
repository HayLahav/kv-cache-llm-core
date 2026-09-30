# 04 — PPA & Synthesis

> Power, performance, and area of the Edge LLM Decode Core. **Every number in this document was
> measured** with the open-source flow in `scripts/synth.tcl`: Yosys 0.69 technology mapping plus
> nextpnr-ecp5 place & route, and cycle counts from Icarus simulation of the same RTL. Nothing here
> comes from Vivado, which was not available in the environment that produced these reports. The
> Vivado branch of `synth.tcl` is provided for users who have it. Figures that are *projections*
> rather than measurements are labelled as such.

---

## 1. Flow and Reproduction

```bash
yosys -c scripts/synth.tcl                                  # N = 4  -> build/synth/
SYNTH_N=8 SYNTH_OUT=build/synth_n8 yosys -c scripts/synth.tcl   # N = 8  -> build/synth_n8/
SYNTH_SEED=5 yosys -c scripts/synth.tcl                     # choose the nextpnr placement seed
vivado -mode batch -source scripts/synth.tcl                # Vivado non-project (WNS, Fmax, util)
```

| Step | Tool / command | Output |
|------|----------------|--------|
| Xilinx 7-series mapping | `synth_xilinx -family xc7 -flatten` (LUT mapping through classic ABC) | `yosys_xc7_util.rpt`: LUT/FF/CARRY4/DSP48E1/RAMB |
| Component breakdown | Same flow on `kv_unpack_dequant` (combinational), `kv_cache_controller`, `pe` | `yosys_xc7_{unpack,kvc,pe}.rpt` |
| Lattice ECP5 mapping | `synth_ecp5` → JSON | `yosys_ecp5_util.rpt` |
| Place & route + STA | `nextpnr-ecp5 --85k --package CABGA756 --speed 8 --freq 200` | `nextpnr_ecp5.log`: routed Fmax and critical path |
| Vivado (optional) | `synth_design -mode out_of_context` → place → route | `vivado_util*.rpt`, `vivado_timing.rpt`, `vivado_summary.txt` (WNS, Fmax) |

> **Tool note.** The Windows build of Yosys 0.69 used here asserts inside ABC9's `write_xaiger2`
> back-end. `synth.tcl` therefore runs the standard `synth_xilinx`/`synth_ecp5` scripts in stages and
> performs the LUT-mapping step with classic `abc -lut`. Every other step is the stock script. The
> result is equivalent for resource estimation and is the reason the script contains
> `synth_xc7`/`synth_ecp5_abc` helpers.

Target clock: 200 MHz (5.0 ns). Device for P&R: **Lattice ECP5-85F, speed grade 8**, a low-cost FPGA
whose EBR block RAM (4.1–4.3 ns clock-to-out without an output register) and combinational
MULT18X18D (3.07 ns) set the ceiling for this design.

---

## 2. Resource Utilisation

### 2.1 Full core

Default configuration: `OUT_LANES = N` (one array row per result beat).

| Resource | Xilinx 7-series (N = 4) | Xilinx 7-series (N = 8) | ECP5-85F (N = 4) | ECP5-85F (N = 8)¹ |
|----------|-------------------------|-------------------------|------------------|------------------|
| LUTs (LUT1–6 / TRELLIS_COMB) | 2 995 | 8 378 | 4 890 | 11 936 |
| LUT as SRL / LUTRAM | 16 SRL16E, 2 RAM32M | 80 SRL16E, 2 RAM32M | 2 RAMW | 2 RAMW |
| Flip-flops | 3 596 | 9 819 | 4 495 | 12 671 |
| Carry (CARRY4) | 198 | 582 | (in COMB) | (in COMB) |
| Multipliers | **16 DSP48E1** | **64 DSP48E1** | **16 MULT18X18D** | **64 MULT18X18D** |
| Block RAM | 2 RAMB36 + 2 RAMB18 | 4 RAMB36 | 6 DP16KD | 8 DP16KD |
| Wide muxes | 266 MUXF7, 93 MUXF8 | 442 MUXF7, 95 MUXF8 | — | — |
| I/O (ECP5 placed) | — | — | 297 / 365 | 297 / 365 |

¹ The ECP5 place-and-route of N = 8 uses `OUT_LANES = 4` (a 128-bit result port). A 256-bit port
would need about 425 pins, more than the 365 I/Os of the largest ECP5-85F package. Embedded in an SoC,
the port never reaches pins and the full width costs only fabric. The Xilinx column is the full
`OUT_LANES = 8` build.

Exactly one DSP per PE, with no LUT multipliers. For reference, an xc7a100t has 63 400 LUTs,
240 DSP48E1, and 135 RAMB36. The 4×4 core uses about 5 % of its LUTs and 7 % of its DSPs. The 8×8
core uses about 13 % of its LUTs and 27 % of its DSPs.

**Cost of the wide result port.** Against the 32-bit port it replaces, `OUT_LANES = N` adds 284 LUTs
at N = 4 (2 711 → 2 995, +10 %) and 1 431 LUTs at N = 8 (6 947 → 8 378, +21 %). The addition is the
N-lane row-select mux on the capture buffer. Flip-flops are unchanged, because the capture buffer
already held the whole tile.

### 2.2 Where the area goes (Xilinx 7-series, N = 4)

| Block | LUTs | FFs | DSP | BRAM | Notes |
|-------|------|-----|-----|------|-------|
| One `pe` | 67 | 87 | 1 | — | 8 CARRY4 for the 32-bit accumulator; 16 × PE ≈ 1 070 LUT / 1 390 FF |
| `kv_cache_controller` | 741 | 604 | — | 2 RAMB36 | Includes the shared INT4 quantiser (≈ 547 LUT, §3.2), the read FSM, and the append pipeline |
| `kv_unpack_dequant` (16 lanes, combinational) | 144 (+32 MUXF7) | 0 | — | — | ≈ 9 LUT per lane, **1.1 LUT per output bit** |
| Top-level glue | remainder | remainder | — | 2 RAMB18 (ping-pong banks) | Tile former (512 FF), capture buffer (512 FF), serializer mux, feed stage, parser |

The flip-flop count is dominated by deliberate pipelining (3 stages per PE plus systolic forwarding
registers) and by two N·N·32-bit / N·d_k·8-bit register files: the capture buffer and the tile
former. §6 lists ways to trade these for BRAM.

---

## 3. INT4 Packing: BRAM Savings vs. LUT Overhead

### 3.1 BRAM savings (measured)

The KV cache holds `R = SINK + W = 68` rows per K and per V. Packing sets the row width.

| Storage | Row width | Xilinx (SDP ≤ 72 b/port) | ECP5 (DP16KD ≤ 36 b/port) |
|---------|-----------|--------------------------|---------------------------|
| INT8 (hypothetical) | 128 b | 2 RAMB36 per memory → **4 RAMB36** | 4 DP16KD per memory → **8 DP16KD** |
| **INT4 packed (this design)** | 64 b | 1 RAMB36 per memory → **2 RAMB36** (measured) | 2 DP16KD per memory → **4 DP16KD** (measured; 6 total incl. 2 ping-pong) |
| Saving | 50 % | **2 RAMB36** | **4 DP16KD** |

At the default size the cache is *width*-bound: 68 rows barely touch a 512-deep BRAM. The
saving is exactly 50 % because each row needs half as many BRAM columns. At realistic LLM sizes
the cache becomes *capacity*-bound, and the saving stays at 50 %. The table below is a
projection from bit counts (`bits / 36 864` per RAMB36):

| Config (per layer, per KV head, K + V) | INT8 bits | INT4 bits | RAMB36 (INT8 → INT4) |
|----------------------------------------|-----------|-----------|----------------------|
| d_k = 16, 4 + 64 tokens (this core) | 34 816 | 17 408 | 4 → 2 (measured) |
| d_k = 64, 4 + 256 tokens | 266 240 | 133 120 | ≥ 8 → ≥ 4 |
| d_k = 128, 4 + 1 024 tokens | 2 105 344 | 1 052 672 | ≥ 58 → ≥ 29 |

### 3.2 LUT overhead (measured)

Sub-byte storage costs logic at both ends:

| Block | Lanes | LUTs | Scaling |
|-------|-------|------|---------|
| Unpacker / dequantiser (read path) | 16 | **144** (+32 MUXF7) | 9 LUT/lane: sign extension is wiring, and the shift is a 5:1 mux per bit |
| Quantiser, one lane per element (K and V in parallel) | 32 | 1 091 | 34 LUT/lane (shift by σ, round bit, saturate) |
| **Quantiser, time-shared K → V (this design)** | 16 | **547** | Half the lanes. One APPEND spans ≥ 3 AXI beats, so sharing costs no throughput |

**Trade-off at the default size.** INT4 costs 144 + 547 ≈ **690 LUTs** (≈ 1.1 % of an xc7a100t)
and saves **2 RAMB36** (≈ 1.5 % of the device's BRAM). It also halves the SRAM bits read per KV
row. Area alone is roughly break-even at d_k = 16. INT4 wins decisively at scale:
* The read-side unpacker scales with d_k (≈ 9 LUT per element). The quantiser can be narrowed
  further, because it runs once per token rather than once per read.
* The BRAM saving scales with d_k × context length × layers × KV heads.

At d_k = 128 with a 1 K window the saving is ~29 RAMB36 per KV head per layer, against roughly
1 150 LUTs for a full-width unpacker.

### 3.3 Dynamic power

Measured power requires a vendor power model with switching activity (for example Vivado
`report_power` with the regression's `.saif`), which is not part of this open-source flow. The
qualitative effects follow from the architecture:

* **KV SRAM reads.** 64-bit instead of 128-bit reads mean half the bit-lines and sense amplifiers
  switch per row. Dynamic read energy per token scales ≈ ½ (docs/01 §5.3). Only the selected
  memory (K *or* V) is enabled per burst, because the read enable is gated by `sel_q`.
* **Off-chip traffic.** For caches spilled to external memory (the realistic case), bytes moved
  are halved. At about 10–100× the energy of an on-chip INT8 MAC per bit, this is the dominant
  energy lever.
* **Operand isolation.** The PE's S0 operand registers load only on valid beats (`if (v_in)`),
  so DSP inputs do not toggle during bubbles, drain, or idle.
* **Reset-free datapath.** Resets only on control state keep the reset network small.

---

## 4. Timing

### 4.1 Final results

Six nextpnr placement seeds per netlist, on the final RTL (`OUT_LANES = N`; 4 for the N = 8 ECP5 run).

| Configuration | Post-route Fmax, seeds 1–6 (ECP5-85F, speed 8) | Critical path class |
|---------------|-----------------------------------------------|---------------------|
| N = 4 | 138.0 / 141.3 / **142.4** / 135.9 / 140.3 / 135.5 MHz (reported run `build/synth`: seed 5, 140.3 MHz) | PE multiplier or KV EBR read path, depending on placement |
| N = 8 | 108.2 / 109.7 / 121.5 / 118.9 / **122.0** / 116.7 MHz (reported run `build/synth_n8`: seed 1, 108.2 MHz) | PE: `a_s0` → MULT18X18D → `prod_s1` (3.46 ns logic, 4.7–5.8 ns routing) |

The wide result port is not on the critical path at either size. The N = 8 results show a wider seed spread and a median about 2 % lower
(117.8 vs. 120.7 MHz) than the 32-bit-port build. That comes from placement congestion: 96 more I/Os and about 1 500 more
LUTs around the same 64-DSP column structure.

The 200 MHz target is **not met on ECP5**. §4.3 shows why the remaining paths are bounded by hard
blocks. On a Xilinx 7-series part the corresponding primitives are substantially faster: BRAM
clock-to-out is ≈ 2 ns, and DSP48E1 has internal A/B/M/P registers that Vivado infers from exactly
the S0/S1/S2 structure of `pe.v`. The Vivado branch of `synth.tcl` reports WNS and Fmax directly
for anyone who can run it. No Xilinx timing number is claimed here.

### 4.2 Timing-closure log

Each row is one iteration: measure the critical path, restructure the RTL, re-run the full
regression (all tests passing after every step), and re-place-and-route.

| # | Critical path found (ns) | Fix applied | Fmax after |
|---|--------------------------|-------------|------------|
| 0 | `cfg_shift` → 9-bit rounding adder → shift → clamp → **KV SRAM write data** (13.3) | — (baseline) | 75.4 MHz |
| 1 | same | Append path pipelined (accept → quantise → write). Rounding rewritten adder-free: `(x>>>s) + x[s−1]` | 86.5 MHz |
| 2 | Ping-pong EBR → bank/mode/zero-pad muxes → **combinational MULT18** (11.6) | Ingress register at the array edge (skew i+1 / j+1) | 101.5 MHz |
| 3 | Quantiser saturation compares mapped to carry chains (9.9) | Saturation detected as plain bit logic: "bits [7:3] ≠ sign" | 115.9 MHz |
| 4 | KV EBR (4.26 clk→q) → K/V select → unpack → register (8.6) | Unpacker `PIPELINE=2`: packed row registered before unpacking | 136.9 MHz |
| 5 | Read FSM `idx == n_win − 1` (16-bit subtract + compare → state CE) | Phase bounds registered at burst start; index narrowed to log2(W) bits | 143.1 MHz |
| 6 | Ping-pong EBR clk→q → muxes → ingress (7.0) | `PP_OUT_REG=1`: per-bank output registers before the bank mux; feed tags delayed 1 cycle | 145.0 MHz |
| 7 | PE register → MULT18 → `prod_s1` | PE S0 operand stage (3-stage PE) | 144.0 MHz |
| 8 | `gap_cnt >= GAP` comparator → issue logic → engine FSM (6.95) | Comparator replaced by a registered `gap_ok` flag (cycle-equivalent) | 147.7 MHz |
| 9 | N = 8: S0 register merged with the forwarding register by `opt_merge`, so one flop drives both the local DSP and the neighbour | Operand isolation (`if (v_in)`) makes S0 logically distinct and also saves power | N = 8: 118 → 119–126 MHz |
| 10 | Quantiser area (1 091 LUT) | Quantiser time-shared K → V, fed from a dedicated input register (`q_x`) so the K/V select sits before a flop | N = 4: 132.6–146.8 MHz across seeds, with ≈ 10 % fewer LUTs (2 997 → 2 711) |
| 11 | Throughput, not timing: the 32-bit result port capped N = 8 at 32 % MAC utilisation (§5.4) | `OUT_LANES = N` result port (one array row per beat) | Final: N = 4 **135.5–142.4 MHz**, N = 8 **108.2–122.0 MHz** across seeds; N = 8 decode step 837 → 453 cycles |

### 4.3 Critical path analysis: KV read → MAC

The KV-read-to-MAC datapath is now a chain of short register-to-register hops. No hop contains
more than about three LUT levels plus one hard block:

```
 stage  from                      logic in between                         to                 worst (ns)
 ─────  ────────────────────────  ───────────────────────────────────────  ─────────────────  ──────────
  R0    idx / win_start regs      log2(W)-bit add, +SINK, phase mux         KV EBR address      ≈ 7.3 *
  R1    KV EBR (no out-reg)       4.26 clk→q + K/V select (1 LUT)           unpack in-reg       ≈ 7.0 *
  R2    unpack in-reg             sext4→8 + ≤4-bit shift mux (1–2 LUT)      unpack out-reg      < 4
  R3    unpack out-reg            SCORE: tile row write / CONTEXT: lane mux tile_row / f_op    < 5
  R4    tile_row regs             d_k:1 column mux per lane (sc_d)          f_op                < 5
  R5    f_op → f2_op              (register copy, aligned with 2-cycle PP)  f2_op               < 2
  R6    f2_op + PP out-reg        mode mux + zero-pad                       array ingress       < 4
  R7    ingress / skew regs       (register chain)                          PE a_s0 / b_s0      < 3
  R8    a_s0 / b_s0               MULT18X18D 3.07 ns (combinational) + routing  prod_s1       7.5–8.5 *
  R9    prod_s1                   sign-extend + 32-bit add (carry chain)    acc / res           < 5
```

`*` marks the paths that trade places as the critical path across seeds. All three are bounded by
ECP5 hard-block timing:

1. **R8, the MAC.** The multiplier is used combinationally because Yosys's ECP5 flow does not pack
   the MULT18X18D input/pipeline registers. The S0/S1 flops exist in RTL but sit in the fabric,
   so each crossing pays a routing hop in and out. This dominates at N = 8, where 64 DSPs spread
   the placement.
2. **R1, the KV EBR.** 4.26 ns clock-to-out without the EBR output register.
3. **R0, the KV read address.** Yosys wraps the inferred memory's read port with extra logic
   around the address. Registering the next read address (computing `raddr` one cycle ahead)
   removes the add from this path.

**Next steps**, not implemented, in order of expected gain:
- Enable the EBR output register (`REGMODE_B = OUTREG`) for the KV SRAMs. The stream protocol
  then needs a 2-deep skid buffer, because the output register no longer doubles as the hold
  register.
- Instantiate `MULT18X18D` with `REG_INPUTA/B_CLK` and `REG_PIPELINE_CLK` enabled, or rely on
  Vivado DSP48 register inference on Xilinx.
- Pre-compute `raddr` one cycle ahead in the read FSM.

---

## 5. Throughput and Bandwidth

### 5.1 Measured cycle counts

Measured in simulation. The context is full (S = 68 = 4 sinks + 64 window), the result port is
always ready, and the operand banks are pre-loaded. The LOAD of the next step's operand is hidden
by the ping-pong buffer.

| N | Result port (`OUT_LANES`) | SCORE (cycles) | CONTEXT (cycles) | Decode step (cycles) | MACs / step | MAC utilisation |
|---|---------------------------|----------------|------------------|----------------------|-------------|-----------------|
| 4 | 1 (32 b) | 408 | 323 | 731 | 8 704 | SCORE 66.7 %, CONTEXT 84.2 %, 74.4 % |
| 4 | **4 (128 b, default)** | 364 | 311 | **675** | 8 704 | SCORE 74.7 %, CONTEXT 87.5 %, **80.6 %** |
| 8 | 1 (32 b) | 604 | 233 | 837 | 17 408 | SCORE 22.5 %, CONTEXT 58.4 %, 32.5 % |
| 8 | 2 (64 b) | 340 | 201 | 541 | 17 408 | SCORE 40.0 %, CONTEXT 67.7 %, 50.3 % |
| 8 | 4 (128 b, ECP5 P&R build) | 268 | 185 | **453** | 17 408 | SCORE 50.7 %, CONTEXT 73.5 %, **60.0 %** |
| 8 | **8 (256 b, default)** | 244 | 177 | **421** | 17 408 | SCORE 55.7 %, CONTEXT 76.8 %, **64.6 %** |

MACs per step = 2 · S · N · d_k (SCORE + CONTEXT). Utilisation = MACs / (cycles × N²).

The lane sweep matches the port analysis in §5.4. At N = 8 the array produces N²/d_k = 4 results
per cycle, so 4 lanes recover almost all of the gain (453 vs. 421 cycles), and 8 lanes add the last
7 % by draining each tile in fewer beats.

### 5.2 Sustained throughput (GOPS)

1 MAC = 2 ops. Clocks are the reported post-route runs (N = 4: 140.3 MHz; N = 8: 108.2 MHz, the
slowest of its six seeds). N = 8 uses the 4-lane cycle count, because that is the configuration
that was placed and routed.

| N | Clock | Peak (N² × 2 × f) | Sustained, decode step | SCORE only | CONTEXT only | Step latency |
|---|-------|-------------------|------------------------|------------|--------------|--------------|
| 4 | 140.3 MHz | 4.49 GOPS | **3.62 GOPS** | 3.35 GOPS | 3.93 GOPS | 4.81 µs |
| 8 | 108.2 MHz | 13.84 GOPS | **8.31 GOPS** | 7.03 GOPS | 10.18 GOPS | 4.19 µs |

Against the previous 32-bit-port build (3.27 and 4.87 GOPS), sustained throughput rises 11 % at N = 4
and **1.7× at N = 8**, despite the lower N = 8 clock. At the best N = 8 seed (122.0 MHz) the same
cycle count gives 9.38 GOPS. With the full 8-lane port (421 cycles; clock not measured, since that
build does not fit the ECP5 package) the cycle count alone gives another 8 %.

A step latency of 4.81 µs gives about 208 000 attention steps per second per KV head for the
4×4 core, over a context that never grows past 68 rows.

### 5.3 Effective memory bandwidth

| N | KV bytes read / step (packed) | Packed read BW | INT8-equivalent BW delivered | Peak KV port (packed / INT8-eq.) |
|---|-------------------------------|----------------|------------------------------|----------------------------------|
| 4 | 544 (K) + 4 × 544 (V, one pass per 4 dims) = 2 720 B | **0.57 GB/s** | **1.13 GB/s** | 1.12 / 2.24 GB/s |
| 8 | 544 + 2 × 544 = 1 632 B | 0.39 GB/s | 0.78 GB/s | 0.87 / 1.73 GB/s |

"INT8-equivalent" is the bandwidth an unpacked INT8 cache would need to deliver the same
elements. The INT4 path supplies twice the logical elements per SRAM bit read. The peak port
rate is one 8-byte packed row per cycle.

### 5.4 Bottlenecks identified by measurement

1. **Result port (fixed).** SCORE emits `S · N` INT32 words, while the array produces `N²/d_k` words
   per cycle of reduction: 1 word/cycle at N = 4 and 4 words/cycle at N = 8. With a 32-bit port the
   N = 8 SCORE took 604 cycles, almost exactly its 544 output words. The port is now `OUT_LANES × 32`
   bits (default one array row per beat), which cut the N = 8 decode step from 837 to 421 cycles.
   On-chip reduction (for example a running max for the softmax) would shrink the output further.
2. **Single-buffered tile former (SCORE).** A tile issues d_k beats and then waits for N new rows.
   Combined with the one-tile-in-flight capture rule, this now dominates SCORE: about 21 cycles per
   tile at N = 4 against an ideal 16. **Fix:** double-buffer `tile_row` (512 more flip-flops at N = 4).
3. **V re-streaming (CONTEXT).** Each of the d_k/N passes re-reads all V rows (4× at N = 4). **Fix:**
   an N-deep V row buffer or a d_k-wide array. The KV read port has headroom (§5.3), so this
   costs bandwidth, not time.

With (2) also fixed, the MAC-bound step time would approach `2 · S · d_k / N` cycles plus drain,
about 272 + 40 cycles at N = 8, or **≈ 1.35× faster than the current 421**. This is a projection,
not a measurement.

---

## 6. Area and Timing Knobs

| Knob | Where | Effect |
|------|-------|--------|
| `PP_OUT_REG` | `llm_decode_top`, `ping_pong_buffer` | 1: +1 cycle feed latency, removes BRAM clk→q from the MAC-feed path (default). 0: fewer flops |
| `kv_unpack_dequant.PIPELINE` | top: 2 | 0 / 1 / 2 register stages. 2 isolates the KV SRAM clock-to-out |
| `N` | top | 4 → 8: 4× DSP, ~2.8× LUT, ~2.7× FF (measured) |
| `OUT_LANES` | top (default N; must divide N) | Result beat width. N = 8: 1 → 4 → 8 lanes gives 837 → 453 → 421 cycles per decode step, for +21 % LUTs at 8 lanes |
| `WINDOW_SIZE`, `SINK_COUNT` | top | Only SRAM depth and a few pointer bits change; the logic is independent of depth |
| Capture buffer / tile former | top | Could move to LUTRAM/BRAM to save ~1 000 FFs, at the cost of a read port on the serializer |
