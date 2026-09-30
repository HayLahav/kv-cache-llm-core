"""
tb_utils.py -- Shared cocotb infrastructure: configuration, clock/reset,
AXI4-Stream source (random tvalid gaps) and sink (random tready backpressure).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cocotb
from cocotb.clock import Clock
from cocotb.queue import Queue
from cocotb.triggers import ReadOnly, RisingEdge


@dataclass(frozen=True)
class CoreConfig:
    n: int
    head_dim: int
    sink_count: int
    window_size: int
    buf_depth: int
    seed: int
    steps: int
    out_lanes: int

    @staticmethod
    def from_env() -> "CoreConfig":
        e = os.environ
        return CoreConfig(
            n=int(e.get("CORE_N", "4")),
            head_dim=int(e.get("CORE_HEAD_DIM", "16")),
            sink_count=int(e.get("CORE_SINK", "4")),
            window_size=int(e.get("CORE_WINDOW", "64")),
            buf_depth=int(e.get("CORE_BUF_DEPTH", "128")),
            seed=int(e.get("CORE_SEED", "20260930")),
            steps=int(e.get("CORE_STEPS", "120")),
            out_lanes=int(e.get("CORE_OUT_LANES", e.get("CORE_N", "4"))),
        )


async def start_clock_and_reset(dut, cycles: int = 5) -> None:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def wait_cycles(dut, n: int) -> None:
    for _ in range(n):
        await RisingEdge(dut.clk)


class AxisSource:
    """Drives s_axis_* with optional random idle cycles between beats."""

    def __init__(self, dut, clk, idle_prob: float = 0.0, rng: Optional[random.Random] = None):
        self.dut = dut
        self.clk = clk
        self.idle_prob = idle_prob
        self.rng = rng or random.Random(0)
        self.beats_sent = 0
        dut.s_axis_tvalid.value = 0
        dut.s_axis_tdata.value = 0
        dut.s_axis_tlast.value = 0

    async def send(self, beats: Sequence[Tuple[int, int]]) -> None:
        for data, last in beats:
            while self.idle_prob and self.rng.random() < self.idle_prob:
                self.dut.s_axis_tvalid.value = 0
                await RisingEdge(self.clk)
            self.dut.s_axis_tdata.value = data
            self.dut.s_axis_tlast.value = last
            self.dut.s_axis_tvalid.value = 1
            while True:
                await ReadOnly()
                accepted = bool(self.dut.s_axis_tready.value)
                await RisingEdge(self.clk)
                if accepted:
                    break
            self.beats_sent += 1
        self.dut.s_axis_tvalid.value = 0
        self.dut.s_axis_tlast.value = 0


class AxisSink:
    """Consumes m_axis_* with random tready; splits each beat into `lanes` signed
    INT32 words (lane k at bits [32k+31:32k]) and groups them into tlast-delimited
    packets."""

    def __init__(self, dut, clk, ready_prob: float = 1.0, rng: Optional[random.Random] = None,
                 lanes: int = 1):
        self.dut = dut
        self.lanes = lanes
        self.clk = clk
        self.ready_prob = ready_prob
        self.rng = rng or random.Random(1)
        self.packets: Queue = Queue()
        self.words_received = 0
        self.stall_cycles = 0
        self._cur: List[int] = []
        dut.m_axis_tready.value = 0
        cocotb.start_soon(self._run())

    async def _run(self) -> None:
        while True:
            ready = 1 if self.rng.random() < self.ready_prob else 0
            self.dut.m_axis_tready.value = ready
            await ReadOnly()
            if int(self.dut.m_axis_tvalid.value):
                if ready:
                    beat = int(self.dut.m_axis_tdata.value)
                    for k in range(self.lanes):
                        w = (beat >> (32 * k)) & 0xFFFFFFFF
                        self._cur.append(w - (1 << 32) if w & 0x80000000 else w)
                    self.words_received += self.lanes
                    if int(self.dut.m_axis_tlast.value):
                        self.packets.put_nowait(self._cur)
                        self._cur = []
                else:
                    self.stall_cycles += 1
            await RisingEdge(self.clk)

    async def recv(self) -> List[int]:
        pkt = await self.packets.get()
        # packets are queued from the ReadOnly phase; step to the next edge so the
        # caller may drive signals again
        await RisingEdge(self.clk)
        return pkt

    def pending_words(self) -> int:
        return len(self._cur)
