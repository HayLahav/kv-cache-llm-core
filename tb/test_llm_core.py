"""
test_llm_core.py -- End-to-end tests of llm_decode_top over AXI4-Stream.

  * test_autoregressive_decode : 120+ decode steps (APPEND -> LOAD Q -> SCORE ->
                                 LOAD P -> CONTEXT) with random tvalid gaps on
                                 the slave and random tready backpressure on
                                 the master; all results bit-exact vs. golden.
  * test_corner_cases          : all-zeros, INT8 saturation (-128/+127) and
                                 alternating polarity, across a window wrap.
  * test_pingpong_prefetch     : both banks pre-loaded, third LOAD must stall
                                 until a bank is released; LOAD beats accepted
                                 while the array computes.
  * test_empty_and_reconfig    : op on empty cache is a no-op; CLEAR + CFG.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List

import cocotb
from cocotb.triggers import ReadOnly, RisingEdge, with_timeout

from golden_model import (
    DecodeCoreModel,
    p_buffer_words,
    pkt_append,
    pkt_cfg,
    pkt_clear,
    pkt_context,
    pkt_load,
    pkt_score,
    q_buffer_words,
    softmax_to_int8,
)
from tb_utils import AxisSink, AxisSource, CoreConfig, start_clock_and_reset, wait_cycles

CFG = CoreConfig.from_env()
N, D = CFG.n, CFG.head_dim


# ------------------------------------------------------------------------------
# Harness
# ------------------------------------------------------------------------------
class Harness:
    def __init__(self, dut, idle_prob: float, ready_prob: float, seed: int):
        self.dut = dut
        self.rng = random.Random(seed)
        self.src = AxisSource(dut, dut.clk, idle_prob, random.Random(seed + 11))
        self.sink = AxisSink(dut, dut.clk, ready_prob, random.Random(seed + 13), CFG.out_lanes)
        self.model = DecodeCoreModel(N, D, CFG.sink_count, CFG.window_size)
        self.expected: deque = deque()
        self.checked_ops = 0
        self.checked_words = 0
        self.overlap_beats = 0      # LOAD beats accepted while the engine was busy
        self.cycles = 0
        self.axi_violations = 0
        cocotb.start_soon(self._monitor())

    async def _monitor(self) -> None:
        """Counts prefetch overlap and checks AXI master stability under backpressure."""
        dut = self.dut
        prev_stalled = False
        prev_data = prev_last = 0
        while True:
            await ReadOnly()
            self.cycles += 1
            if int(dut.pp_wr_en.value) and int(dut.e_state.value) != 0:
                self.overlap_beats += 1
            tvalid = int(dut.m_axis_tvalid.value)
            tready = int(dut.m_axis_tready.value)
            if prev_stalled:
                data = int(dut.m_axis_tdata.value)
                last = int(dut.m_axis_tlast.value)
                if not tvalid or data != prev_data or last != prev_last:
                    self.axi_violations += 1
            prev_stalled = bool(tvalid and not tready)
            if prev_stalled:
                prev_data = int(dut.m_axis_tdata.value)
                prev_last = int(dut.m_axis_tlast.value)
            await RisingEdge(dut.clk)

    # ---- command helpers (update the golden model in lock-step) ----
    async def cfg(self, shift: int) -> None:
        self.model.cfg(shift)
        await self.src.send(pkt_cfg(shift))

    async def clear(self) -> None:
        self.model.clear()
        await self.src.send(pkt_clear())

    async def append(self, k: List[int], v: List[int]) -> None:
        self.model.append(k, v)
        await self.src.send(pkt_append(k, v))

    async def score(self, q_heads: List[List[int]], tag: str) -> List[int]:
        exp = self.model.score(q_heads)
        await self.src.send(pkt_load(q_buffer_words(q_heads)))
        await self.src.send(pkt_score())
        if exp:
            self.expected.append((tag, exp))
        return exp

    async def context(self, p_heads: List[List[int]], tag: str) -> List[int]:
        exp = self.model.context(p_heads)
        await self.src.send(pkt_load(p_buffer_words(p_heads)))
        await self.src.send(pkt_context())
        if exp:
            self.expected.append((tag, exp))
        return exp

    async def check_all(self, timeout_us: int = 20000) -> None:
        while self.expected:
            tag, exp = self.expected.popleft()
            got = await with_timeout(self.sink.recv(), timeout_us, "us")
            assert len(got) == len(exp), f"{tag}: got {len(got)} words, expected {len(exp)}"
            if got != exp:
                bad = [i for i, (g, e) in enumerate(zip(got, exp)) if g != e]
                i = bad[0]
                raise AssertionError(
                    f"{tag}: {len(bad)} mismatches, first at word {i}: got {got[i]} exp {exp[i]}"
                )
            self.checked_ops += 1
            self.checked_words += len(exp)

    async def checker(self) -> None:
        """Background checker: drains expected results as they arrive."""
        while True:
            if self.expected:
                await self.check_all()
            else:
                await RisingEdge(self.dut.clk)


def rand_row(rng: random.Random, lo: int = -128, hi: int = 127) -> List[int]:
    return [rng.randint(lo, hi) for _ in range(D)]


def probs_from_scores(scores: List[int], seq: int) -> List[List[int]]:
    """Host-side softmax per head over the SCORE stream -> INT8 P[h][t]."""
    per_head = [[scores[t * N + h] for t in range(seq)] for h in range(N)]
    return [softmax_to_int8(s, D, scale=1.0 / 64.0) for s in per_head]


async def reset(dut) -> None:
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tdata.value = 0
    dut.s_axis_tlast.value = 0
    dut.m_axis_tready.value = 0
    await start_clock_and_reset(dut)


# ------------------------------------------------------------------------------
@cocotb.test()
async def test_autoregressive_decode(dut):
    """Auto-regressive decode loop with random backpressure on both AXI interfaces."""
    await reset(dut)
    h = Harness(dut, idle_prob=0.25, ready_prob=0.55, seed=CFG.seed)
    chk = cocotb.start_soon(h.checker())

    await h.cfg(4)  # full-range INT8 -> INT4 scaling
    steps = max(CFG.steps, 100)
    for t in range(steps):
        if t == steps // 2:
            await h.cfg(3)  # dynamic re-scaling mid-sequence
        await h.append(rand_row(h.rng), rand_row(h.rng))
        q = [rand_row(h.rng) for _ in range(N)]
        scores = await h.score(q, f"step{t}/SCORE")
        seq = h.model.context_len
        p = probs_from_scores(scores, seq)
        await h.context(p, f"step{t}/CONTEXT")
        if t % 20 == 0:
            dut._log.info(f"step {t:3d}: context_len={seq:3d} tokens_total={h.model.kv.token_count}")

    await h.check_all()
    chk.cancel()
    assert h.axi_violations == 0, f"{h.axi_violations} AXI master stability violations"
    assert h.sink.stall_cycles > 0, "backpressure was never exercised"
    assert h.overlap_beats > 0, "ping-pong prefetch never overlapped compute"
    await ReadOnly()
    assert int(dut.token_count.value) == steps
    dut._log.info(
        f"{steps} decode steps, {h.checked_ops} ops, {h.checked_words} INT32 words bit-exact; "
        f"{h.sink.stall_cycles} backpressure stalls; {h.overlap_beats} LOAD beats overlapped "
        f"with compute; {h.cycles} cycles"
    )


@cocotb.test()
async def test_corner_cases(dut):
    """All zeros, saturation extremes and alternating polarity, across window wrap-around."""
    await reset(dut)
    h = Harness(dut, idle_prob=0.1, ready_prob=0.7, seed=CFG.seed + 1)
    chk = cocotb.start_soon(h.checker())
    n_tok = CFG.sink_count + CFG.window_size + 9  # force the window to wrap

    patterns = {
        "zeros":       (lambda t, e: 0,                        lambda h_, d: 0,    0),
        "max_pos":     (lambda t, e: 127,                      lambda h_, d: 127,  127),
        "max_neg":     (lambda t, e: -128,                     lambda h_, d: -128, 127),
        "mixed_sat":   (lambda t, e: -128,                     lambda h_, d: 127,  127),
        "alternating": (lambda t, e: 127 if (t + e) % 2 else -128,
                        lambda h_, d: -128 if (h_ + d) % 2 else 127, 127),
    }
    for name, (kv_fn, q_fn, p_val) in patterns.items():
        for shift in (0, 4):
            await h.clear()
            await h.cfg(shift)
            for t in range(n_tok):
                row = [kv_fn(t, e) for e in range(D)]
                await h.append(row, row)
            q = [[q_fn(hh, d) for d in range(D)] for hh in range(N)]
            await h.score(q, f"{name}/s{shift}/SCORE")
            seq = h.model.context_len
            p = [[p_val if p_val == 0 else (p_val if (hh + tt) % 3 else -128)
                  for tt in range(seq)] for hh in range(N)]
            await h.context(p, f"{name}/s{shift}/CONTEXT")
            await h.check_all()
            dut._log.info(f"corner '{name}' shift={shift}: PASS")
    chk.cancel()
    assert h.axi_violations == 0


@cocotb.test()
async def test_pingpong_prefetch(dut):
    """Two banks fill back-to-back; a third LOAD stalls until compute releases a bank."""
    await reset(dut)
    h = Harness(dut, idle_prob=0.0, ready_prob=0.0, seed=CFG.seed + 2)  # sink stalled
    rng = h.rng
    for _ in range(CFG.sink_count + 20):
        await h.append(rand_row(rng), rand_row(rng))

    qs = [[rand_row(rng) for _ in range(N)] for _ in range(3)]
    exp = [h.model.score(q) for q in qs]

    # bank A and bank B are loaded before any compute
    await h.src.send(pkt_load(q_buffer_words(qs[0])))
    await h.src.send(pkt_load(q_buffer_words(qs[1])))
    await ReadOnly()
    assert int(dut.pp_bank_state.value) == 0b1010, "both banks should be READY"
    await RisingEdge(dut.clk)

    # SCORE #1 acquires bank A. The result stream is stalled, so the engine
    # cannot finish feeding tiles and bank A stays in COMPUTE.
    await h.src.send(pkt_score())
    # third LOAD: header accepted, payload must stall (no EMPTY bank)
    third = cocotb.start_soon(h.src.send(pkt_load(q_buffer_words(qs[2]))))
    await wait_cycles(dut, 60)
    assert not third.done(), "third LOAD should be back-pressured by the ping-pong buffer"
    await ReadOnly()
    assert int(dut.s_axis_tready.value) == 0
    assert int(dut.busy.value) == 1
    await RisingEdge(dut.clk)

    # release the result stream: bank A is freed as soon as SCORE #1 has fed its
    # last tile, and LOAD #3 streams into it while the engine is still busy
    h.sink.ready_prob = 0.5
    await with_timeout(third, 50, "us")
    await with_timeout(h.src.send(pkt_score() + pkt_score()), 50, "us")
    for i in range(3):
        got = await with_timeout(h.sink.recv(), 50, "us")
        assert got == exp[i], f"prefetched SCORE #{i} mismatch"
    assert h.overlap_beats > 0, "no LOAD beat overlapped compute"
    dut._log.info(f"ping-pong: 3 prefetched SCOREs bit-exact, {h.overlap_beats} overlapped beats")


@cocotb.test()
async def test_empty_and_reconfig(dut):
    """SCORE on an empty cache is a no-op; CLEAR resets sequence state; CFG clamps shift."""
    await reset(dut)
    h = Harness(dut, idle_prob=0.0, ready_prob=1.0, seed=CFG.seed + 3)
    q = [rand_row(h.rng) for _ in range(N)]
    assert h.model.score(q) == []
    await h.src.send(pkt_score())            # no bank loaded, empty cache -> no-op
    await wait_cycles(dut, 10)
    await ReadOnly()
    assert int(dut.busy.value) == 0 and int(dut.m_axis_tvalid.value) == 0
    await RisingEdge(dut.clk)

    await h.cfg(7)                            # clamps to 4 in both RTL and model
    for t in range(3):
        await h.append(rand_row(h.rng), rand_row(h.rng))
    await h.score(q, "after-empty/SCORE")
    await h.check_all(timeout_us=100)

    await h.clear()
    await h.append(rand_row(h.rng), rand_row(h.rng))
    await h.score(q, "after-clear/SCORE")     # single-token partial tile
    p = [[h.rng.randint(-128, 127)] for _ in range(N)]
    await h.context(p, "after-clear/CONTEXT")  # single-token reduction (K = 1)
    await h.check_all(timeout_us=100)
    await ReadOnly()
    assert int(dut.token_count.value) == 1
