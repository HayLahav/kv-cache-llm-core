# 03 — Verification Guide

> How the design is verified: the cocotb ↔ simulator bridge, the bit-exact golden model, every
> test and what it proves, how to get and read waveforms, and the coverage/assertion matrix.
> Every claim here is backed by a test in `tb/` that passes in the current regression.

---

## 1. Environment

| Component       | Version used           | Notes                                                        |
|-----------------|------------------------|--------------------------------------------------------------|
| Icarus Verilog  | 14.0 (devel)           | From the OSS CAD Suite bundle. Compiled with `-g2012` by the cocotb runner; the RTL itself is Verilog-2001 |
| cocotb          | 2.1.0                  | `pip install -r requirements.txt`                            |
| Python          | 3.12 (CPython)         | Any CPython ≥ 3.9 supported by cocotb works                  |
| GTKWave         | OSS CAD Suite bundle   | `make wave`                                                  |

The tests are launched by `tb/run_tests.py`, which uses cocotb's Python runner API
(`cocotb_tools.runner`). No `make`-based cocotb Makefiles are involved, so the same command works in
Linux shells, macOS, Windows PowerShell, and Git Bash:

```bash
python tb/run_tests.py kv            # KV cache unit tests
python tb/run_tests.py top           # end-to-end, 4x4
python tb/run_tests.py top8          # end-to-end, 8x8
python tb/run_tests.py top_w1        # end-to-end, 4x4 with a 32-bit (1-lane) result port
python tb/run_tests.py all --waves   # everything + VCDs
python tb/run_tests.py top --testcase test_corner_cases   # one test
```

`make sim_kv`, `make sim_top`, `make sim_top8`, and `make sim` are thin wrappers around the same
runner.

---

## 2. The Cocotb-to-RTL Bridge (VPI)

```
 ┌──────────────────────────── Python process (embedded) ────────────────────────────┐
 │  test_*.py coroutines ──► cocotb scheduler ──► handles: dut.s_axis_tdata.value = x │
 │        ▲    await RisingEdge / ReadOnly / Timer                     │               │
 └────────┼────────────────────────────────────────────────────────────┼──────────────┘
          │ callbacks (cbValueChange, cbReadOnlySynch, cbAfterDelay)   │ vpi_put_value /
          │                                                            │ vpi_get_value
 ┌────────┴────────────────────────────────────────────────────────────▼──────────────┐
 │ libcocotbvpi_icarus.vpl  (GPI layer: VPI implementation for Icarus)                 │
 ├─────────────────────────────────────────────────────────────────────────────────────┤
 │ vvp  sim.vvp    (Icarus runtime, elaborated llm_decode_top + vcd_dump)              │
 └─────────────────────────────────────────────────────────────────────────────────────┘
```

1. `iverilog` compiles the RTL (plus `vcd_dump.v` when waves are requested) into `sim.vvp`.
   Parameters are overridden with `-P` (for example `N=8`) from `run_tests.py`.
2. `vvp -m libcocotbvpi_icarus sim.vvp` loads cocotb's **VPI** library. At start-of-simulation the
   library embeds a Python interpreter, imports the test module named in `COCOTB_TEST_MODULES`, and
   hands each `@cocotb.test()` coroutine to the scheduler.
3. Signal access goes through **VPI handles**. `dut.u_kv.rd_state` walks the design hierarchy with
   `vpi_handle_by_name`, which is how the tests probe internal state such as the collision guard
   and bank states. Writes are `vpi_put_value` (applied in the inertial/NBA region). Reads are
   `vpi_get_value`.
4. `await RisingEdge(dut.clk)` registers a `cbValueChange` callback. `await ReadOnly()` registers
   `cbReadOnlySynch`, which runs after all events of the timestep have settled.

**Sampling discipline used throughout `tb/`.** Inputs are driven right after a `RisingEdge`. DUT
outputs are sampled in `ReadOnly()` of the same cycle, which gives the settled pre-edge values
that the next clock edge will act on. A handshake is recorded exactly when `valid && ready` holds
in `ReadOnly()`. Driving signals from `ReadOnly()` is illegal. For that reason `AxisSink.recv()`
steps to the next edge before returning: packets are queued from the ReadOnly phase.
`AxisSource`, `AxisSink`, and the monitors all follow this rule, which makes the testbench free of
races by construction.

---

## 3. The Golden Model (`tb/golden_model.py`)

The model is pure Python and bit-exact to the RTL. Its self-test runs with `python
tb/golden_model.py`.

| Function / class          | Mirrors RTL                          | Exactness guarantee                                   |
|---------------------------|--------------------------------------|-------------------------------------------------------|
| `quant4(x, s)`            | `kv_cache_controller.quant4`         | `clamp(floor((x + 2^(s−1)) / 2^s), −8, 7)`, `s` clamped to 4. Python `>>` is a floor shift, like Verilog `>>>` |
| `dequant4(q, s)`          | `kv_unpack_dequant` lane             | `sext4→8` then `<< s` with 8-bit wrap                 |
| `pack_int4_row`           | packer (element e → bits [4e+3:4e])  | identical bit layout                                  |
| `unpack_int4_row`         | unpacker                             | identical bit layout                                  |
| `KVCacheModel`            | sink/window pointers + read FSM      | same `sink_fill`, `win_fill`, `head_ptr`, slot mapping, and **Phase A / Phase B read order**. It also tracks token IDs per slot |
| `KVCacheModel.expected_token_ids()` | *specification*, not RTL   | StreamingLLM semantics computed independently of addresses: `[0..S_s−1] ∪ [max(S_s, t−W) .. t−1]` |
| `gemm_int32`              | systolic array                       | Python big-int accumulation, **every partial sum asserted to fit INT32**, so no overflow can hide |
| `score_expected` / `context_expected` | output serializer order  | token-major (SCORE) and pass-major/head-major (CONTEXT) |
| `DecodeCoreModel`         | whole core at the command level      | updated in lock-step with every AXI packet the test sends |
| `pkt_*`, `q_buffer_words`, `p_buffer_words` | host protocol      | header/beat encoding and Q^T / P^T bank layouts       |

The model has two layers on purpose. `read_order()` mirrors the hardware addressing, and
`expected_token_ids()` restates the *algorithmic* definition of attention sinks. The tests assert
that the two agree, so a bug shared by the RTL and the address model would still be caught by the
spec-level check.

---

## 4. Test Suites

### 4.1 `tb/test_kv_cache.py`

Top level `tb/kv_cache_tb_top.v`: the controller, a `PIPELINE=2` unpacker on its read stream
(the exact topology used in the core), and a free-standing combinational unpacker.

| Test | What it drives | What it proves |
|------|----------------|----------------|
| `test_unpack_exhaustive` | All 16 nibble codes in **every lane** (16 rotations) × shifts 0…7 → 2048 lane checks | Bit-exact sign extension of −8…+7 and dequantisation; shifts > 4 clamp to 4 |
| `test_int4_roundtrip` | For σ = 0…4: a row holding every INT4 value, INT8 saturation rows (±127/−128), zeros, random rows, and **all 256 INT8 inputs** | Quantise → pack → SRAM → unpack is bit-exact vs. golden for K and V. Rounding and clamp corners included |
| `test_sink_retention_wraparound` | 207 tokens (window wraps 3+ times) with token IDs encoded in the nibbles. Burst reads at 9 checkpoints including W−1, W, and W+1, with **40 % random backpressure** | Sinks never change after prefill. The window streams oldest-first. `m_phase` = 0 for exactly `sink_fill` rows then 1. Status counters match every cycle. Read order equals the spec-level token list |
| `test_clear_and_refill` | CLEAR mid-sequence, burst on an empty cache, refill | Empty burst emits nothing; a new sequence restarts sinks at slot 0 |
| *(monitor)* collision guard | Every cycle of every burst | `app_ready` is never high while the read FSM is active (`violations == 0`) |

### 4.2 `tb/test_llm_core.py`

Top level `llm_decode_top`, driven only through AXI4-Stream (plus read-only internal probes).

| Test | Stimulus | Checks |
|------|----------|--------|
| `test_autoregressive_decode` | **120 decode steps**. Each step: APPEND (random INT8 K,V) → LOAD Q^T → SCORE → host softmax on the *golden* scores → LOAD P^T → CONTEXT. σ = 4, switched to 3 at step 60. Slave `tvalid` idles 25 % of the time; master `tready` is high only 55 % of the time | 240 ops, **31 208 INT32 words bit-exact** (4×4). The context grows 1 → 68 and then stays at 68 while the window wraps. Backpressure stalls observed (> 6 000; > 25 000 on `top_w1`). LOAD beats accepted while the engine is busy (> 5 800 overlapped beats = prefetch) |
| `test_corner_cases` | 77 tokens (forces a wrap) for each pattern × σ ∈ {0, 4}: **all zeros**, **max positive** (+127), **max negative** (−128), **mixed saturation** (K = −128 against Q = +127), **alternating polarity** (±127/−128 checkerboard over tokens and lanes); P patterns include −128 | Bit-exact SCORE and CONTEXT, including the most negative INT4 × INT8 products and the largest accumulations |
| `test_pingpong_prefetch` | Two LOADs fill both banks before any compute (`bank_state = READY/READY`). SCORE #1 starts while the result port is stalled. A third LOAD is issued | The third LOAD's payload is **back-pressured** (`s_axis_tready = 0`) while no bank is free. When the result port opens, bank A is released and LOAD #3 streams in while the engine is still busy. All three SCOREs are bit-exact and use the correct bank |
| `test_empty_and_reconfig` | SCORE on an empty cache; CFG with σ = 7; CLEAR; 1-token SCORE and CONTEXT | The no-op returns to idle with no output. σ clamps to 4 in both RTL and model. A partial tile with 1 valid row and a K = 1 reduction are exact |
| *(monitor)* AXI master stability | Every cycle | While `tvalid && !tready`, `tdata`/`tlast` must not change and `tvalid` must stay high (`axi_violations == 0`) |

The `top8` target re-runs the whole suite with `N = 8` (2 CONTEXT passes, 8-row tiles, 256-bit result
beats), producing 62 416 bit-exact words in the decode test. The `top_w1` target re-runs it with
`OUT_LANES = 1` so the narrow-port serializer stays verified. `AxisSink` splits every beat into
`CORE_OUT_LANES` signed INT32 words (lane 0 first) before comparison, so the same checks cover
every port width.

### 4.3 Regression results (current RTL)

| Target | Tests | Result | Sim time | Notes |
|--------|-------|--------|----------|-------|
| `kv`   | 4     | 4/4 PASS | 23 µs  | 2048 unpack lane checks, 207-token wraparound |
| `top`  | 4     | 4/4 PASS | 752 µs | 31 208 words bit-exact, 0 AXI violations (128-bit result beats) |
| `top8` | 4     | 4/4 PASS | 528 µs | 62 416 words bit-exact (256-bit result beats) |
| `top_w1` | 4   | 4/4 PASS | 972 µs | 31 208 words bit-exact through a 32-bit result port |

### 4.4 Verifying the verification (mutation checks)

The testbench was checked against deliberately broken RTL. Each mutant must fail:

| Mutation | Failing tests |
|----------|---------------|
| Unpacker zero-extends instead of sign-extending (`{4'b0000, nib}`) | `test_unpack_exhaustive`, `test_int4_roundtrip`, `test_sink_retention_wraparound` |
| Window read address loses its `& (W−1)` wrap mask | `test_sink_retention_wraparound` (only visible once the window has wrapped, as expected) |

---

## 5. Waveforms (VCD + GTKWave)

```bash
make wave                  # = python tb/run_tests.py top --waves ; gtkwave sim_build/top/dump.vcd scripts/llm_decode_top.gtkw
make wave_kv               # KV cache harness
python tb/run_tests.py top --waves --testcase test_empty_and_reconfig   # small, fast VCD
```

How it works: with `--waves`, `run_tests.py` compiles `tb/vcd_dump.v` as a second root module
(`iverilog -s vcd_dump`) with `-DDUMP_TOP=<toplevel>`. That module calls
`$dumpfile("dump.vcd"); $dumpvars(0, <toplevel>)`. The runner is also switched from `vvp -none` to
`vvp -vcd`, so the file is a plain **VCD**, written to `sim_build/<target>/dump.vcd`.

`scripts/llm_decode_top.gtkw` pre-loads a grouped signal view:

| Group | Signals | What to look for |
|-------|---------|------------------|
| AXI4-Stream slave | `s_axis_*`, `p_state` | Header → payload sequencing; `tready` dropping in P_LOAD when both banks are full |
| KV cache | `token_count`, `sink_fill`, `win_fill`, `head_ptr`, `rd_state`, `kv_m_*` | `rd_state` 1 → 2 (sink → window) and `kv_m_phase` flipping mid-burst; `head_ptr` wrapping modulo 64 in the slot address |
| INT4 unpack | `u_valid/u_ready/u_data` | Sign-extended lanes (for example `0xF8` = −8 at σ = 0, `0x80` = −128 at σ = 4) |
| Ping-pong | `pp_bank_state`, `pp_wr_bank`, `pp_rd_bank`, `pp_wr_en`, `pp_rd_en` | `pp_wr_en` pulses while `e_state` = RUN (prefetch overlap); `bank_state` nibbles 0 → 1 → 2 → 3 → 0 |
| Engine / array | `e_state`, `sc_d`, `e_pass`, `tile_full`, `feed_go`, `a_valid/a_first/a_last`, `arr_done`, `cap_full_q`, `pending` | `a_last` → `arr_done` = 10 cycles (N = 4). The next tile's feed overlaps the previous tile's drain |
| AXI4-Stream master | `m_axis_*` (hex; one INT32 per 32-bit lane) | `tvalid` held under backpressure; `tlast` on the final beat of each op |

Tips: in GTKWave, right-click a bus and choose *Data Format → Signed Decimal* for INT8/INT32 lanes.
The full 120-step regression produces a very large VCD, so the single-testcase command above is
better for interactive exploration.

---

## 6. Coverage Matrix

| Feature / requirement | Directed test(s) | Random / soak | Assertion / checker |
|-----------------------|------------------|---------------|---------------------|
| INT8 × INT8 → INT32 MAC, signed | corner_cases (±127, −128) | autoregressive_decode | golden compare; INT32 range assert in model |
| Systolic skew + zero padding | partial tiles (1–3 rows) in every decode step < 4 tokens; `after-clear` 1-row tile | decode (S mod 4 ≠ 0) | golden compare |
| Tile overlap / capture hazard | `test_empty_and_reconfig` (K = 1 CONTEXT) | decode with random backpressure (hold path) | golden compare (a clobbered result would mismatch) |
| INT4 sign extension, all 16 codes | unpack_exhaustive | — | lane-by-lane compare |
| Quantiser rounding / clamp, all 256 inputs × 5 shifts | int4_roundtrip | decode (σ = 4 → 3) | packed-word compare |
| Attention sinks immutable | sink_retention_wraparound | decode (120 > 68 tokens) | sink rows compared at every checkpoint |
| Window wrap (`& (W−1)`) | checkpoints at W−1, W, W+1, 2W+7, 3W+11 | decode | read order = spec token list |
| Two-phase burst ordering | phase-flag check | — | `m_phase` sequence assert |
| KV read/write collision prevention | collision guard monitor | all KV tests | `violations == 0` |
| Ping-pong bank switching | pingpong_prefetch | decode (overlap counter) | `bank_state == READY/READY`; `tready == 0` on the 3rd LOAD |
| AXI slave flow control | random `tvalid` gaps (25 %) | decode, corner | protocol-level driver |
| AXI master backpressure | random `tready` (45 % stall) | decode, corner | stability monitor (`axi_violations == 0`) |
| `tlast` framing | every op | every op | packet length == expected length |
| CFG / CLEAR / empty-cache no-op | empty_and_reconfig | decode (CFG change) | `busy == 0`, no output |
| Parameter scaling (N = 8) | `top8` target | `top8` decode | full suite |
| Result-port width (`OUT_LANES` = N and 1) | `top`, `top8`, `top_w1` targets | all decode tests | per-lane compare, stability monitor on the full beat |

### Assertion checklist

- [x] Every SCORE/CONTEXT result word equals the golden INT32 value (with no tolerance)
- [x] Result packet length equals `S·N` (SCORE) or `N·d_k` (CONTEXT), with `tlast` on the last word only
- [x] Golden model: every partial sum fits INT32
- [x] Sink rows are bit-identical before and after any number of window wraps
- [x] Stream order equals `sinks ++ window (oldest → newest)` and equals the spec-level token list
- [x] `sink_fill`, `win_fill`, `head_ptr`, `token_count` match the model after every append
- [x] `app_ready = 0` whenever the KV read FSM is active
- [x] AXI master holds `tdata/tlast/tvalid` stable while stalled
- [x] A third outstanding LOAD is back-pressured until a bank is released
- [x] Prefetch overlap observed (> 0 LOAD beats accepted while the engine is busy)
- [x] Backpressure actually exercised (> 0 stall cycles)
