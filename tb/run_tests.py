#!/usr/bin/env python3
"""
run_tests.py -- Build and run the cocotb regressions on Icarus Verilog.

    python tb/run_tests.py kv            # KV cache unit tests
    python tb/run_tests.py top           # end-to-end core, 4x4 array
    python tb/run_tests.py top8          # end-to-end core, 8x8 array
    python tb/run_tests.py top_w1        # end-to-end core, 4x4, 32-bit result port
    python tb/run_tests.py all --waves   # everything, dump VCDs

Waveforms (with --waves) land in sim_build/<target>/dump.vcd.
Environment overrides: CORE_SEED, CORE_STEPS.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cocotb_tools.runner import Icarus

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "rtl"
TB = ROOT / "tb"

RTL_SOURCES = [
    RTL / "pe.v",
    RTL / "systolic_array.v",
    RTL / "ping_pong_buffer.v",
    RTL / "kv_unpack_dequant.v",
    RTL / "kv_cache_controller.v",
    RTL / "llm_decode_top.v",
]

class IcarusVcd(Icarus):
    """Icarus runner that emits a real VCD (vvp -vcd) instead of -none / -fst."""

    vcd = False

    def _test_command(self):
        cmds = super()._test_command()
        if self.vcd:
            cmds = [["-vcd" if a in ("-none", "-fst") else a for a in cmd] for cmd in cmds]
        return cmds


TARGETS = {
    "kv": dict(
        toplevel="kv_cache_tb_top",
        module="test_kv_cache",
        extra_sources=[TB / "kv_cache_tb_top.v"],
        params=dict(HEAD_DIM=16, SINK_COUNT=4, WINDOW_SIZE=64),
        env=dict(CORE_N="4", CORE_HEAD_DIM="16", CORE_SINK="4", CORE_WINDOW="64"),
    ),
    "top": dict(
        toplevel="llm_decode_top",
        module="test_llm_core",
        extra_sources=[],
        params=dict(N=4, HEAD_DIM=16, SINK_COUNT=4, WINDOW_SIZE=64, BUF_DEPTH=128, BUF_AW=7),
        env=dict(CORE_N="4", CORE_HEAD_DIM="16", CORE_SINK="4", CORE_WINDOW="64",
                 CORE_BUF_DEPTH="128"),
    ),
    "top8": dict(
        toplevel="llm_decode_top",
        module="test_llm_core",
        extra_sources=[],
        params=dict(N=8, HEAD_DIM=16, SINK_COUNT=4, WINDOW_SIZE=64, BUF_DEPTH=128, BUF_AW=7),
        env=dict(CORE_N="8", CORE_HEAD_DIM="16", CORE_SINK="4", CORE_WINDOW="64",
                 CORE_BUF_DEPTH="128"),
    ),
    # narrow (one INT32 per beat) result port, kept under regression
    "top_w1": dict(
        toplevel="llm_decode_top",
        module="test_llm_core",
        extra_sources=[],
        params=dict(N=4, HEAD_DIM=16, SINK_COUNT=4, WINDOW_SIZE=64, BUF_DEPTH=128, BUF_AW=7,
                    OUT_LANES=1),
        env=dict(CORE_N="4", CORE_HEAD_DIM="16", CORE_SINK="4", CORE_WINDOW="64",
                 CORE_BUF_DEPTH="128", CORE_OUT_LANES="1"),
    ),
}


def run_target(name: str, waves: bool, testcase: str | None) -> bool:
    cfg = TARGETS[name]
    build_dir = ROOT / "sim_build" / name
    sources = RTL_SOURCES + cfg["extra_sources"]
    build_args = []
    defines = {}
    if waves:
        sources = sources + [TB / "vcd_dump.v"]
        defines["DUMP_TOP"] = cfg["toplevel"]
        build_args += ["-s", "vcd_dump"]

    env = dict(cfg["env"])
    for key in ("CORE_SEED", "CORE_STEPS"):
        if key in os.environ:
            env[key] = os.environ[key]

    runner = IcarusVcd()
    runner.vcd = waves
    runner.build(
        sources=sources,
        hdl_toplevel=cfg["toplevel"],
        parameters=cfg["params"],
        defines=defines,
        build_args=build_args,
        build_dir=build_dir,
        timescale=("1ns", "1ps"),
        always=True,
    )
    results = runner.test(
        hdl_toplevel=cfg["toplevel"],
        test_module=cfg["module"],
        test_dir=build_dir,
        build_dir=build_dir,
        extra_env=env,
        testcase=testcase,
        test_args=[],
    )
    return _all_passed(results)


def _all_passed(results_xml) -> bool:
    import xml.etree.ElementTree as ET

    tree = ET.parse(results_xml)
    failures = 0
    total = 0
    for case in tree.iter("testcase"):
        total += 1
        if case.find("failure") is not None or case.find("error") is not None:
            failures += 1
    print(f"[run_tests] {results_xml}: {total - failures}/{total} passed")
    return total > 0 and failures == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("target", choices=list(TARGETS) + ["all"])
    ap.add_argument("--waves", action="store_true", help="dump sim_build/<target>/dump.vcd")
    ap.add_argument("--testcase", default=None, help="run a single cocotb test by name")
    args = ap.parse_args()

    sys.path.insert(0, str(TB))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(TB)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
    )

    names = list(TARGETS) if args.target == "all" else [args.target]
    ok = True
    for name in names:
        ok &= run_target(name, args.waves, args.testcase)
    print("[run_tests] OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
