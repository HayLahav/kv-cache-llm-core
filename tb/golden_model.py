"""
golden_model.py -- Bit-exact Python reference model of the Edge LLM Decode Core.

Everything here mirrors the RTL exactly:
  * INT8 -> INT4 quantisation (round-half-up arithmetic shift, clamp [-8, 7])
  * nibble packing (element e at bits [4e+3:4e]) and sign-extend / shift unpacking
  * Attention-sink + power-of-two circular window cache addressing and readout order
  * INT32 GEMM results (Python ints, range-checked so an overflow can never hide)
  * AXI4-Stream command encoding and result ordering of llm_decode_top
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# ------------------------------------------------------------------------------
# Opcodes (must match llm_decode_top.v)
# ------------------------------------------------------------------------------
OP_CFG = 0x0
OP_APPEND = 0x1
OP_LOAD = 0x2
OP_SCORE = 0x3
OP_CONTEXT = 0x4
OP_CLEAR = 0x5

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


# ------------------------------------------------------------------------------
# Scalar helpers
# ------------------------------------------------------------------------------
def to_signed(value: int, bits: int) -> int:
    """Interpret the low `bits` of value as a two's-complement integer."""
    value &= (1 << bits) - 1
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def to_unsigned(value: int, bits: int) -> int:
    return value & ((1 << bits) - 1)


def clamp_shift(shift: int) -> int:
    return 4 if shift > 4 else shift


def quant4(x: int, shift: int) -> int:
    """INT8 -> INT4 exactly as kv_cache_controller.quant4 (returns signed int)."""
    assert -128 <= x <= 127
    s = clamp_shift(shift)
    rnd = 0 if s == 0 else 1 << (s - 1)
    t = (x + rnd) >> s  # Python >> on ints is an arithmetic (floor) shift
    return max(-8, min(7, t))


def dequant4(q: int, shift: int) -> int:
    """INT4 -> INT8 exactly as kv_unpack_dequant: sext4to8 then <<shift (8-bit wrap)."""
    assert -8 <= q <= 7
    return to_signed(to_unsigned(q, 8) << clamp_shift(shift), 8)


def unpack_nibble(nib: int, shift: int) -> int:
    return dequant4(to_signed(nib, 4), shift)


# ------------------------------------------------------------------------------
# Row packing
# ------------------------------------------------------------------------------
def pack_int4_row(q_values: Sequence[int]) -> int:
    """Pack signed INT4 values into one word, element e at bits [4e+3:4e]."""
    word = 0
    for e, q in enumerate(q_values):
        assert -8 <= q <= 7
        word |= to_unsigned(q, 4) << (4 * e)
    return word


def unpack_int4_row(word: int, head_dim: int, shift: int) -> List[int]:
    return [unpack_nibble((word >> (4 * e)) & 0xF, shift) for e in range(head_dim)]


def quantise_row(x_values: Sequence[int], shift: int) -> List[int]:
    return [quant4(x, shift) for x in x_values]


def pack_int8_lanes(values: Sequence[int], lane_bits: int = 8) -> int:
    """Pack signed values into lanes (lane i at bits [i*lane_bits +: lane_bits])."""
    word = 0
    for i, v in enumerate(values):
        word |= to_unsigned(v, lane_bits) << (lane_bits * i)
    return word


def unpack_int8_lanes(word: int, count: int, lane_bits: int = 8) -> List[int]:
    return [to_signed(word >> (lane_bits * i), lane_bits) for i in range(count)]


# ------------------------------------------------------------------------------
# KV cache model (attention sinks + circular window)
# ------------------------------------------------------------------------------
@dataclass
class KVCacheModel:
    head_dim: int = 16
    sink_count: int = 4
    window_size: int = 64
    k_mem: List[Optional[int]] = field(default_factory=list)
    v_mem: List[Optional[int]] = field(default_factory=list)
    tok_mem: List[Optional[int]] = field(default_factory=list)  # token id per slot
    token_count: int = 0
    sink_fill: int = 0
    win_fill: int = 0
    head_ptr: int = 0

    def __post_init__(self) -> None:
        assert self.window_size & (self.window_size - 1) == 0, "WINDOW_SIZE must be 2^k"
        depth = self.sink_count + self.window_size
        self.k_mem = [None] * depth
        self.v_mem = [None] * depth
        self.tok_mem = [None] * depth

    def clear(self) -> None:
        self.token_count = self.sink_fill = self.win_fill = self.head_ptr = 0

    def append(self, k_int8: Sequence[int], v_int8: Sequence[int], shift: int) -> int:
        """Quantise, pack and store one token. Returns the physical slot written."""
        if self.sink_fill < self.sink_count:
            addr = self.sink_fill
            self.sink_fill += 1
        else:
            addr = self.sink_count + (self.head_ptr & (self.window_size - 1))
            self.head_ptr += 1
            self.win_fill = min(self.win_fill + 1, self.window_size)
        self.k_mem[addr] = pack_int4_row(quantise_row(k_int8, shift))
        self.v_mem[addr] = pack_int4_row(quantise_row(v_int8, shift))
        self.tok_mem[addr] = self.token_count
        self.token_count += 1
        return addr

    def read_order(self) -> List[int]:
        """Physical slots in stream order: Phase A sinks, Phase B window oldest-first."""
        order = list(range(self.sink_fill))
        start = (self.head_ptr - self.win_fill) & (self.window_size - 1)
        for i in range(self.win_fill):
            order.append(self.sink_count + ((start + i) & (self.window_size - 1)))
        return order

    def context_token_ids(self) -> List[int]:
        return [self.tok_mem[a] for a in self.read_order()]

    def expected_token_ids(self) -> List[int]:
        """Attention-sink semantics, independent of addressing: sinks + last W tokens."""
        t = self.token_count
        sinks = list(range(min(t, self.sink_count)))
        win_lo = max(self.sink_count, t - self.window_size)
        return sinks + list(range(win_lo, t))

    def packed_rows(self, sel_v: bool) -> List[int]:
        mem = self.v_mem if sel_v else self.k_mem
        return [mem[a] for a in self.read_order()]

    def unpacked_rows(self, sel_v: bool, shift: int) -> List[List[int]]:
        return [unpack_int4_row(w, self.head_dim, shift) for w in self.packed_rows(sel_v)]


# ------------------------------------------------------------------------------
# Integer GEMM (strict INT32 accumulation)
# ------------------------------------------------------------------------------
def _check_int32(v: int) -> int:
    assert INT32_MIN <= v <= INT32_MAX, f"INT32 overflow: {v}"
    return v


def gemm_int32(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> List[List[int]]:
    """C = A @ B with every partial sum range-checked against INT32."""
    rows, inner, cols = len(a), len(b), len(b[0])
    out = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        assert len(a[i]) == inner
        for j in range(cols):
            acc = 0
            for k in range(inner):
                acc = _check_int32(acc + a[i][k] * b[k][j])
            out[i][j] = acc
    return out


def transpose(m: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(r) for r in zip(*m)]


def score_expected(k_rows: List[List[int]], q_heads: List[List[int]]) -> List[int]:
    """SCORE output stream: for each token (stream order), heads 0..N-1."""
    s = gemm_int32(k_rows, transpose(q_heads))  # [S][N]
    return [v for row in s for v in row]


def context_expected(p_heads: List[List[int]], v_rows: List[List[int]], n: int) -> List[int]:
    """CONTEXT output stream: for each pass p, heads i, dims p*N+j."""
    o = gemm_int32(p_heads, v_rows)  # [N][D]
    head_dim = len(v_rows[0])
    out = []
    for p in range(head_dim // n):
        for i in range(len(p_heads)):
            for j in range(n):
                out.append(o[i][p * n + j])
    return out


# ------------------------------------------------------------------------------
# Host-side helpers (softmax is done off-array, as in the architecture)
# ------------------------------------------------------------------------------
def softmax_to_int8(scores: Sequence[int], head_dim: int, scale: float = 1.0) -> List[int]:
    """Host softmax over INT32 scores -> INT8 probabilities in [0, 127]."""
    if not scores:
        return []
    z = [s * scale / math.sqrt(head_dim) for s in scores]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    tot = sum(e)
    return [min(127, int(round(127.0 * v / tot))) for v in e]


# ------------------------------------------------------------------------------
# AXI4-Stream packet builders for llm_decode_top
# ------------------------------------------------------------------------------
def header(op: int, arg: int = 0) -> int:
    return (op & 0xF) | ((arg & 0x7) << 8)


def pkt_cfg(shift: int) -> List[tuple]:
    return [(header(OP_CFG, shift), 1)]


def pkt_clear() -> List[tuple]:
    return [(header(OP_CLEAR), 1)]


def pkt_append(k_int8: Sequence[int], v_int8: Sequence[int]) -> List[tuple]:
    return [(header(OP_APPEND), 0), (pack_int8_lanes(k_int8), 0), (pack_int8_lanes(v_int8), 1)]


def pkt_load(words: Sequence[int]) -> List[tuple]:
    assert len(words) >= 1
    beats = [(header(OP_LOAD), 0)]
    for i, w in enumerate(words):
        beats.append((w, 1 if i == len(words) - 1 else 0))
    return beats


def pkt_score() -> List[tuple]:
    return [(header(OP_SCORE), 1)]


def pkt_context() -> List[tuple]:
    return [(header(OP_CONTEXT), 1)]


def q_buffer_words(q_heads: List[List[int]]) -> List[int]:
    """Q^T layout: bank word d = {Q[N-1][d], ..., Q[0][d]}."""
    head_dim = len(q_heads[0])
    return [pack_int8_lanes([q[d] for q in q_heads]) for d in range(head_dim)]


def p_buffer_words(p_heads: List[List[int]]) -> List[int]:
    """P^T layout: bank word t = {P[N-1][t], ..., P[0][t]}."""
    seq = len(p_heads[0])
    return [pack_int8_lanes([p[t] for p in p_heads]) for t in range(seq)]


# ------------------------------------------------------------------------------
# Command-level model of the full core
# ------------------------------------------------------------------------------
class DecodeCoreModel:
    def __init__(self, n: int = 4, head_dim: int = 16, sink_count: int = 4, window_size: int = 64):
        self.n = n
        self.head_dim = head_dim
        self.shift = 0
        self.kv = KVCacheModel(head_dim, sink_count, window_size)

    def cfg(self, shift: int) -> None:
        self.shift = clamp_shift(shift)

    def clear(self) -> None:
        self.kv.clear()

    def append(self, k_int8: Sequence[int], v_int8: Sequence[int]) -> None:
        self.kv.append(k_int8, v_int8, self.shift)

    @property
    def context_len(self) -> int:
        return len(self.kv.read_order())

    def score(self, q_heads: List[List[int]]) -> List[int]:
        if self.kv.token_count == 0:
            return []
        return score_expected(self.kv.unpacked_rows(False, self.shift), q_heads)

    def context(self, p_heads: List[List[int]]) -> List[int]:
        if self.kv.token_count == 0:
            return []
        return context_expected(p_heads, self.kv.unpacked_rows(True, self.shift), self.n)


# ------------------------------------------------------------------------------
# Self-test (python tb/golden_model.py)
# ------------------------------------------------------------------------------
def _self_test() -> None:
    # INT4 round-trip is lossless at shift 0 for every representable value
    for q in range(-8, 8):
        assert quant4(q, 0) == q and dequant4(q, 0) == q
    # full-range scaling corners
    assert quant4(127, 4) == 7 and dequant4(7, 4) == 112
    assert quant4(-128, 4) == -8 and dequant4(-8, 4) == -128
    # packing is little-endian by element
    assert pack_int4_row([1, -1]) == 0xF1
    assert unpack_int4_row(0xF1, 2, 0) == [1, -1]
    # sink retention across wrap-around
    kv = KVCacheModel(4, 4, 8)
    for t in range(50):
        kv.append([t & 7] * 4, [0] * 4, 0)
        assert kv.context_token_ids() == kv.expected_token_ids()
    assert kv.context_token_ids()[:4] == [0, 1, 2, 3]
    print("golden_model self-test: PASS")


if __name__ == "__main__":
    _self_test()
