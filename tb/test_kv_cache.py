"""
test_kv_cache.py -- Unit tests for kv_cache_controller + kv_unpack_dequant
(toplevel: tb/kv_cache_tb_top.v)

  1. Exhaustive unpacker check: every INT4 value in every lane, every shift.
  2. INT4 quantise -> pack -> SRAM -> unpack round trip, bit-exact vs. golden.
  3. Attention-sink retention while the window wraps many times.
  4. Read-stream backpressure, two-phase ordering (phase flags), collision guard.
  5. CLEAR starts a fresh sequence.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import cocotb
from cocotb.triggers import ReadOnly, RisingEdge, Timer

from golden_model import (
    KVCacheModel,
    dequant4,
    pack_int8_lanes,
    quant4,
    to_signed,
    unpack_int8_lanes,
    unpack_nibble,
)
from tb_utils import CoreConfig, start_clock_and_reset, wait_cycles

CFG = CoreConfig.from_env()
D = CFG.head_dim


def init_inputs(dut) -> None:
    dut.clear.value = 0
    dut.cfg_shift.value = 0
    dut.app_valid.value = 0
    dut.app_k.value = 0
    dut.app_v.value = 0
    dut.rd_start.value = 0
    dut.rd_sel.value = 0
    dut.u_ready.value = 0
    dut.raw_packed.value = 0
    dut.raw_shift.value = 0


async def setup(dut) -> None:
    init_inputs(dut)
    await start_clock_and_reset(dut)


async def append(dut, k: List[int], v: List[int]) -> None:
    dut.app_k.value = pack_int8_lanes(k)
    dut.app_v.value = pack_int8_lanes(v)
    dut.app_valid.value = 1
    while True:
        await ReadOnly()
        ok = bool(dut.app_ready.value)
        await RisingEdge(dut.clk)
        if ok:
            break
    dut.app_valid.value = 0


async def read_burst(
    dut, sel_v: bool, rng: random.Random, ready_prob: float = 1.0
) -> Tuple[List[List[int]], List[int], List[int], int]:
    """Launch a burst and collect (unpacked rows, packed rows, phases, app_ready violations)."""
    dut.rd_sel.value = int(sel_v)
    dut.rd_start.value = 1
    await RisingEdge(dut.clk)
    dut.rd_start.value = 0

    rows: List[List[int]] = []
    packed: List[int] = []
    phases: List[int] = []
    violations = 0
    guard = 0
    while True:
        ready = 1 if rng.random() < ready_prob else 0
        dut.u_ready.value = ready
        await ReadOnly()
        # packed-side monitor (controller -> unpacker handshake)
        if int(dut.pk_valid.value) and int(dut.pk_ready.value):
            packed.append(int(dut.pk_data.value))
            phases.append(int(dut.pk_phase.value))
        # collision guard: the write port must be closed while the read FSM is active
        if int(dut.u_kv.rd_state.value) != 0 and int(dut.app_ready.value):
            violations += 1
        done = False
        if int(dut.u_valid.value) and ready:
            rows.append(unpack_int8_lanes(int(dut.u_data.value), D))
            done = bool(int(dut.u_last.value))
        await RisingEdge(dut.clk)
        guard += 1
        assert guard < 20000, "read burst never completed"
        if done:
            break
    dut.u_ready.value = 0
    return rows, packed, phases, violations


# ------------------------------------------------------------------------------
@cocotb.test()
async def test_unpack_exhaustive(dut):
    """Every INT4 value in every lane, for shifts 0..7 (4..7 must clamp to 4)."""
    init_inputs(dut)
    checked = 0
    for shift in range(8):
        dut.raw_shift.value = shift
        for base in range(16):
            nibbles = [(base + e) & 0xF for e in range(D)]
            word = 0
            for e, nb in enumerate(nibbles):
                word |= nb << (4 * e)
            dut.raw_packed.value = word
            await Timer(1, unit="ns")
            got = unpack_int8_lanes(int(dut.raw_unpacked.value), D)
            exp = [unpack_nibble(nb, shift) for nb in nibbles]
            assert got == exp, f"shift={shift} base={base}: got {got} exp {exp}"
            checked += D
    # explicit corner values the spec calls out
    assert dequant4(-8, 0) == -8 and dequant4(7, 0) == 7
    assert dequant4(-8, 4) == -128 and dequant4(7, 4) == 112
    dut._log.info(f"unpacker: {checked} lane checks over all 16 INT4 codes x 8 shifts: PASS")


@cocotb.test()
async def test_int4_roundtrip(dut):
    """Quantise/pack on append and unpack on read are bit-exact for all shifts."""
    await setup(dut)
    rng = random.Random(CFG.seed)
    for shift in range(5):
        dut.clear.value = 1
        await RisingEdge(dut.clk)
        dut.clear.value = 0
        dut.cfg_shift.value = shift
        model = KVCacheModel(D, CFG.sink_count, CFG.window_size)

        rows_k: List[List[int]] = [
            [(e % 16) - 8 for e in range(D)],                  # every INT4 value
            [127 if e % 2 else -128 for e in range(D)],        # INT8 saturation
            [0] * D,
        ]
        rows_k += [[rng.randint(-128, 127) for _ in range(D)] for _ in range(5)]
        # every INT8 input value at least once
        all_vals = list(range(-128, 128))
        rows_k += [all_vals[i:i + D] for i in range(0, 256, D)]
        for k in rows_k:
            v = [-x if x > -128 else 127 for x in k]
            await append(dut, k, v)
            model.append(k, v, shift)

        for sel_v in (False, True):
            rows, packed, _, _ = await read_burst(dut, sel_v, rng)
            assert packed == model.packed_rows(sel_v), f"packed mismatch shift={shift}"
            assert rows == model.unpacked_rows(sel_v, shift), f"unpack mismatch shift={shift}"
        # spot-check the scalar quantiser against the stored nibbles
        for x in (-128, -9, -8, -1, 0, 1, 7, 8, 127):
            q = quant4(x, shift)
            assert -8 <= q <= 7
        dut._log.info(f"shift={shift}: {len(rows_k)} tokens round-tripped bit-exact")


@cocotb.test()
async def test_sink_retention_wraparound(dut):
    """Sinks survive while the circular window wraps > 3 times; order stays chronological."""
    await setup(dut)
    rng = random.Random(CFG.seed + 1)
    model = KVCacheModel(D, CFG.sink_count, CFG.window_size)
    total = CFG.sink_count + 3 * CFG.window_size + 11
    checkpoints = {1, 2, CFG.sink_count, CFG.sink_count + 1,
                   CFG.sink_count + CFG.window_size - 1,
                   CFG.sink_count + CFG.window_size,
                   CFG.sink_count + CFG.window_size + 1,
                   CFG.sink_count + 2 * CFG.window_size + 7, total}
    sink_rows_k = None
    for t in range(1, total + 1):
        tok = t - 1
        # token id is encoded in the INT4 nibbles (shift 0 keeps values exact)
        k = [to_signed((tok >> (4 * (e % 4))) & 0xF, 4) if e < 4 else rng.randint(-8, 7)
             for e in range(D)]
        v = [rng.randint(-8, 7) for _ in range(D)]
        await append(dut, k, v)
        model.append(k, v, 0)

        await ReadOnly()  # sample the post-edge (settled) status registers
        assert int(dut.token_count.value) == model.token_count
        assert int(dut.sink_fill.value) == model.sink_fill
        assert int(dut.win_fill.value) == model.win_fill
        assert int(dut.head_ptr.value) == model.head_ptr
        await RisingEdge(dut.clk)

        if t in checkpoints:
            assert model.context_token_ids() == model.expected_token_ids()
            for sel_v in (False, True):
                rows, packed, phases, viol = await read_burst(dut, sel_v, rng, ready_prob=0.6)
                assert viol == 0, "append port open during a read burst"
                assert packed == model.packed_rows(sel_v), f"t={t} sel_v={sel_v} order mismatch"
                assert rows == model.unpacked_rows(sel_v, 0)
                n_sink = model.sink_fill
                assert phases == [0] * n_sink + [1] * model.win_fill, f"phase flags t={t}"
            if t >= CFG.sink_count:
                cur_sinks = model.packed_rows(False)[:CFG.sink_count]
                if sink_rows_k is None:
                    sink_rows_k = cur_sinks
                assert cur_sinks == sink_rows_k, "sink rows changed after prefill"
            dut._log.info(f"t={t:4d}: context={model.context_token_ids()[:6]}... "
                          f"len={len(model.read_order())} OK")
    assert model.context_token_ids()[:CFG.sink_count] == list(range(CFG.sink_count))


@cocotb.test()
async def test_clear_and_refill(dut):
    """CLEAR empties the cache; an empty burst produces no rows; refill works."""
    await setup(dut)
    rng = random.Random(CFG.seed + 2)
    model = KVCacheModel(D, CFG.sink_count, CFG.window_size)
    for _ in range(CFG.sink_count + 10):
        k = [rng.randint(-8, 7) for _ in range(D)]
        await append(dut, k, k)
        model.append(k, k, 0)
    dut.clear.value = 1
    await RisingEdge(dut.clk)
    dut.clear.value = 0
    model.clear()
    await ReadOnly()
    assert int(dut.token_count.value) == 0
    await RisingEdge(dut.clk)

    # a burst on an empty cache must not produce data
    dut.rd_start.value = 1
    await RisingEdge(dut.clk)
    dut.rd_start.value = 0
    dut.u_ready.value = 1
    for _ in range(10):
        await ReadOnly()
        assert not int(dut.u_valid.value)
        await RisingEdge(dut.clk)

    for _ in range(3):
        k = [rng.randint(-8, 7) for _ in range(D)]
        await append(dut, k, k)
        model.append(k, k, 0)
    rows, packed, phases, _ = await read_burst(dut, False, rng)
    assert packed == model.packed_rows(False)
    assert phases == [0, 0, 0]
    await wait_cycles(dut, 2)
