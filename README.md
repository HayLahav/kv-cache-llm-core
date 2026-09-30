# Edge LLM Decode Core

**A synthesizable Verilog-2001 accelerator for the auto-regressive decode phase of LLM attention,
built around the memory subsystem rather than the MAC array.**

Decode is limited by memory bandwidth, not compute. Every generated token re-reads the whole
KV cache for a handful of multiply-accumulates. This core combines an INT8 systolic array with a
KV-cache controller that attacks the **memory wall** from two sides:

| Lever | Mechanism in RTL | Effect |
|-------|------------------|--------|
| **Bound the cache** | Attention Sinks (4 pinned prompt tokens) + power-of-2 circular sliding window (64 tokens), `slot = SINK + (head_ptr & (W−1))` | Fixed 68-row cache, so decoding can run forever with no out-of-memory stall and no re-allocation. Softmax stays stable because the sink tokens keep their share of the normaliser (StreamingLLM) |
| **Shrink every byte** | Sub-byte **INT4** storage (two signed nibbles per byte), hardware quantise-on-write and sign-extend/dequantise-on-read | 50 % KV storage and read bandwidth, 2× attention arithmetic intensity, half the SRAM bit-lines toggled per row |
| **Hide the external memory** | Ping-pong (double-buffered) operand banks | The next step's Q/P operands stream in while the array computes |

Everything is verified bit-exactly against a Python golden model with cocotb, and taken through
synthesis and place & route with open-source tools.

---

## Architecture at a Glance

```
             s_axis (d_k×8 b)                                          m_axis (N×32 b INT32 row)
                  │                                                               ▲
         ┌────────▼────────┐  LOAD (prefetch)   ┌──────────────────────┐          │
         │ Command parser  ├───────────────────►│ ping_pong_buffer     │          │
         │ CFG APPEND LOAD │                    │ Bank A ◄─► Bank B    │          │
         │ SCORE CONTEXT   │                    └──────────┬───────────┘          │
         │ CLEAR           │                               │ Qᵀ / Pᵀ column       │
         └──┬──────────┬───┘                               ▼                      │
     APPEND │          │ SCORE/CONTEXT          ┌──────────────────────┐   ┌──────┴──────┐
   ┌────────▼──────────▼─────────┐              │ feed: operand align, │   │ capture +   │
   │ kv_cache_controller         │              │ first/last tagging   │   │ serializer  │
   │ INT8→INT4 quantise + pack   │              └──────────┬───────────┘   └──────▲──────┘
   │ K SRAM 68×64b │ V SRAM      │                         ▼                      │
   │ [S0..S3 | circular W=64]    │              ┌──────────────────────┐          │
   │ Read FSM: sinks → window    ├──► unpack ──►│ systolic_array N×N   ├──────────┘
   └─────────────────────────────┘   INT4→INT8  │ INT8×INT8→INT32 PEs  │
                                     sext ≪ σ   └──────────────────────┘
```

* **SCORE** computes `S[τ][h] = Σ_d K[τ][d]·Q[h][d]`, the attention logits of N grouped-query heads
  over the whole cached context (N tokens × N heads per tile).
* **CONTEXT** computes `O[h][c] = Σ_τ P[h][τ]·V[τ][c]`, the attention-weighted value sum
  (N heads × N dims per pass).
* The softmax runs on the host between the two (a scalar, low-bandwidth, non-linear step), so
  the array stays a pure, bit-exact integer engine.

Dataflow for one decode step:
`APPEND(k_t, v_t) → LOAD Qᵀ → SCORE → host softmax → LOAD Pᵀ → CONTEXT`.
The LOAD for the next operand overlaps the current compute through the ping-pong banks.

---

## Repository Layout

```
rtl/                      synthesizable Verilog-2001
  pe.v                    INT8 MAC, 3-stage pipeline, operand isolation, INT32 accumulator
  systolic_array.v        N×N output-stationary mesh, ingress + skew registers, zero padding
  ping_pong_buffer.v      dual-bank SRAM, EMPTY/FILLING/READY/COMPUTE bank FSM
  kv_unpack_dequant.v     packed INT4×2 → sign-extended, shifted INT8 lanes
  kv_cache_controller.v   sink/window addressing, INT4 quantise+pack, two-phase read FSM
  llm_decode_top.v        AXI4-Stream wrapper, parser, compute engine, capture/serializer
tb/
  golden_model.py         bit-exact reference (quantiser, packing, cache, INT32 GEMM, protocol)
  test_kv_cache.py        KV cache + unpacker unit tests
  test_llm_core.py        end-to-end AXI tests (decode loop, corners, ping-pong, reconfig)
  tb_utils.py             AXI4-Stream source/sink with random valid/ready
  kv_cache_tb_top.v       KV test harness;  vcd_dump.v  optional VCD dumper
  run_tests.py            cocotb runner (no make required)
scripts/
  synth.tcl               Yosys (xc7 + ECP5 + nextpnr) / Vivado non-project synthesis
  llm_decode_top.gtkw     GTKWave signal layout
docs/                     design documentation (see the index below)
Makefile                  sim_kv / sim_top / sim_top8 / wave / synth / clean
```

---

## Quickstart

### 1. Tools

| Tool | Purpose | Install |
|------|---------|---------|
| Icarus Verilog ≥ 12, Yosys, nextpnr-ecp5, GTKWave | simulation, synthesis, P&R, waves | Easiest: one [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build/releases) download (Linux/macOS/Windows) |
| Python ≥ 3.9 + cocotb 2.x | testbench | `python -m pip install -r requirements.txt` |
| GNU make (optional) | shortcuts | Every target is a one-line command you can run directly |

```bash
# Linux / macOS
source /path/to/oss-cad-suite/environment
python -m pip install -r requirements.txt
```

```powershell
# Windows PowerShell (use a regular CPython for cocotb; the suite supplies iverilog/yosys)
. C:\path\to\oss-cad-suite\environment.ps1
python -m pip install -r requirements.txt
```

### 2. Run the tests

```bash
python tb/golden_model.py            # golden-model self-test
python tb/run_tests.py kv            # KV cache unit tests          (make sim_kv)
python tb/run_tests.py top           # end-to-end, 4x4 array        (make sim_top)
python tb/run_tests.py top8          # end-to-end, 8x8 array        (make sim_top8)
python tb/run_tests.py top_w1        # end-to-end, 32-bit result port (make sim_top_w1)
python tb/run_tests.py all           # everything                   (make sim)
```

Expected tail:

```
[run_tests] .../sim_build/kv/results.xml: 4/4 passed
[run_tests] .../sim_build/top/results.xml: 4/4 passed
[run_tests] .../sim_build/top8/results.xml: 4/4 passed
[run_tests] .../sim_build/top_w1/results.xml: 4/4 passed
[run_tests] OVERALL: PASS
```

Knobs: `CORE_SEED=<n>` changes the random stimulus, and `CORE_STEPS=<n>` sets the decode-loop length
(minimum 100).

### 3. Open waveforms

```bash
make wave
# or, without make:
python tb/run_tests.py top --waves --testcase test_empty_and_reconfig
gtkwave sim_build/top/dump.vcd scripts/llm_decode_top.gtkw
```

### 4. Synthesize

```bash
yosys -c scripts/synth.tcl                         # make synth   (N=4)
SYNTH_N=8 SYNTH_OUT=build/synth_n8 yosys -c scripts/synth.tcl   # make synth8
vivado -mode batch -source scripts/synth.tcl       # make vivado  (if Vivado is installed)
```

Reports land in `build/synth/`: `yosys_xc7_*.rpt` (Xilinx 7-series resources),
`yosys_ecp5_util.rpt`, and `nextpnr_ecp5.log` (placed-and-routed Fmax and critical path).

---

## Results

### Verification (all passing)

| Suite | Tests | Highlights |
|-------|-------|------------|
| KV cache (`kv`) | 4/4 | All 16 INT4 codes × all lanes × 8 shift settings; all 256 INT8 inputs quantised × 5 scales; sinks unchanged across 3+ window wraps; read/write collision guard never violated |
| Core 4×4 (`top`) | 4/4 | 120 decode steps, **31 208 INT32 results bit-exact** under random `tvalid` gaps and 45 % `tready` stalls; saturation / zero / alternating-polarity corners; ping-pong backpressure |
| Core 8×8 (`top8`) | 4/4 | Same suite, **62 416 results bit-exact** |
| Core 4×4, 32-bit result port (`top_w1`) | 4/4 | Same suite through the narrow port (`OUT_LANES = 1`) |

### Implementation

See [`docs/04_ppa_and_synthesis.md`](docs/04_ppa_and_synthesis.md) for the full tables. All figures
are from Yosys/nextpnr runs of this RTL.

| Metric | 4×4 core | 8×8 core |
|--------|----------|----------|
| Multipliers | 16 DSP48E1 / 16 MULT18 | 64 DSP48E1 / 64 MULT18 |
| LUT / FF (Xilinx 7-series, Yosys) | 2 995 / 3 596 | 8 378 / 9 819 |
| Block RAM (Xilinx / ECP5) | 2 RAMB36 + 2 RAMB18 / 6 DP16KD | 4 RAMB36 / 8 DP16KD |
| Post-route Fmax, ECP5-85F speed 8 (nextpnr, 6 seeds) | 135.5–142.4 MHz | 108.2–122.0 MHz¹ |
| Decode step, full 68-token context (measured cycles) | 675 cycles, 81 % MAC utilisation | 421 cycles, 65 % (453 / 60 % with the 4-lane port used for P&R) |
| Sustained throughput at the reported Fmax | 3.62 GOPS (peak 4.49) | 8.31 GOPS (peak 13.84), 4-lane P&R build |
| INT4 saving | KV cache: 2 RAMB36 instead of 4 (−50 %), for ≈ 690 LUTs of quantiser + unpacker | same |

The result port is `OUT_LANES × 32` bits, default one array row (N INT32s) per beat. Widening it
from 32 bits halved the 8×8 decode step (837 → 421 cycles) for +21 % LUTs.

¹ The ECP5 place-and-route of the 8×8 core uses a 4-lane (128-bit) result port, because a 256-bit
port needs more pins than the ECP5-85F package has.

The 200 MHz target is not met on ECP5. The remaining critical paths are bounded by ECP5 hard blocks
(EBR clock-to-out and a combinational MULT18), and docs/04 §4 has the closure log and next steps.
Vivado was not available when these reports were produced; `make vivado` runs the provided
non-project flow.

---

## Documentation Index

| Document | Contents |
|----------|----------|
| [01 — System Architecture](docs/01_system_architecture.md) | Single-query attention math, GQA mapping, arithmetic-intensity proof and roofline (why decode is memory-bound), attention-sink softmax stability, INT4 quantisation and packing, block diagrams, host protocol |
| [02 — Microarchitecture & RTL](docs/02_microarchitecture_and_rtl.md) | Module hierarchy, signal dictionaries, parser/engine FSMs, tile-overlap rules, **measured** cycle-by-cycle timeline, synthesis guidelines (latches, resets, memory collisions) |
| [03 — Verification Guide](docs/03_verification_guide.md) | cocotb ↔ VPI bridge, golden model, every test and what it proves, VCD/GTKWave guide, coverage matrix, assertion checklist, mutation checks |
| [04 — PPA & Synthesis](docs/04_ppa_and_synthesis.md) | Resource utilisation, INT4 BRAM savings vs. unpacker/quantiser LUT cost, critical-path analysis and the timing-closure log, sustained GOPS and effective GB/s, scaling to 8×8 |

---

## Design Parameters

| Parameter | Default | Constraint |
|-----------|---------|------------|
| `N` | 4 | Array is N×N; tested at 4 and 8 |
| `HEAD_DIM` | 16 | Multiple of N, even |
| `SINK_COUNT` | 4 | ≥ 1 |
| `WINDOW_SIZE` | 64 | Power of two, ≥ 2 |
| `BUF_DEPTH` / `BUF_AW` | 128 / 7 | ≥ max(HEAD_DIM, SINK_COUNT + WINDOW_SIZE) |
| `PP_OUT_REG` | 1 | Ping-pong BRAM output register (timing) |
| `OUT_LANES` | N | INT32 results per m_axis beat; must divide N |
