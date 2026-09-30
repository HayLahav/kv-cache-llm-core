# ------------------------------------------------------------------------------
# synth.tcl -- Non-project synthesis for the Edge LLM Decode Core.
#
# The same script runs under two tools; it detects which one is executing it:
#
#   Yosys  (open source):  yosys -c scripts/synth.tcl
#       * Xilinx 7-series tech mapping (synth_xilinx)  -> LUT/FF/DSP48/BRAM counts
#       * Lattice ECP5 tech mapping  (synth_ecp5)      -> JSON netlist, then
#         nextpnr-ecp5 place & route                   -> real Fmax / slack
#       * Component breakdown: INT4 unpacker LUT cost, KV SRAM mapping
#
#   Vivado (non-project):  vivado -mode batch -source scripts/synth.tcl
#       * synth_design (out-of-context) + opt/place/route on an Artix-7
#       * report_utilization, report_timing_summary (WNS/TNS/WHS), Fmax
#
# Knobs (environment variables):
#   SYNTH_N        array size N          (default 4)
#   SYNTH_PERIOD   target clock, ns      (default 5.0 -> 200 MHz)
#   SYNTH_PART     Vivado part           (default xc7a100tcsg324-1)
#   SYNTH_OUT      report directory      (default build/synth)
#   SYNTH_PNR      1 = run nextpnr-ecp5 after Yosys (default 1)
#   SYNTH_FLOW     Yosys only: all | xc7 | ecp5  (default all)
#   SYNTH_SEED     nextpnr placement seed (default 1)
#   SYNTH_OUT_LANES  INT32 results per m_axis beat (default N). The ECP5 P&R run
#                  uses min(OUT_LANES, 4): at N = 8 a 256-bit port would need more
#                  pins than the largest ECP5-85F package (365 I/O) provides.
# ------------------------------------------------------------------------------

proc env_or {name default} {
    if {[info exists ::env($name)] && $::env($name) ne ""} { return $::env($name) }
    return $default
}

set script_dir [file dirname [file normalize [info script]]]
set root       [file dirname $script_dir]
set N          [env_or SYNTH_N 4]
set PERIOD     [env_or SYNTH_PERIOD 5.0]
set PART       [env_or SYNTH_PART xc7a100tcsg324-1]
set OUT        [file normalize [env_or SYNTH_OUT [file join $root build synth]]]
set RUN_PNR    [env_or SYNTH_PNR 1]
set FLOW       [env_or SYNTH_FLOW all]
set SEED       [env_or SYNTH_SEED 1]
set LANES      [env_or SYNTH_OUT_LANES $N]
set LANES_PNR  [expr {$LANES > 4 ? 4 : $LANES}]
set TOP        llm_decode_top
set FREQ_MHZ   [expr {1000.0 / $PERIOD}]

set RTL [list \
    [file join $root rtl pe.v] \
    [file join $root rtl systolic_array.v] \
    [file join $root rtl ping_pong_buffer.v] \
    [file join $root rtl kv_unpack_dequant.v] \
    [file join $root rtl kv_cache_controller.v] \
    [file join $root rtl llm_decode_top.v] ]

file mkdir $OUT

# ==============================================================================
# Vivado flow
# ==============================================================================
if {[llength [info commands synth_design]] > 0} {
    puts "== Vivado non-project flow: part $PART, N=$N, period ${PERIOD}ns =="
    foreach f $RTL { read_verilog $f }

    synth_design -top $TOP -part $PART -mode out_of_context \
                 -generic N=$N -generic OUT_LANES=$LANES -flatten_hierarchy rebuilt
    create_clock -name clk -period $PERIOD [get_ports clk]
    set_property HD.CLK_SRC BUFGCTRL_X0Y0 [get_ports clk]

    report_utilization -file [file join $OUT vivado_util_synth.rpt]
    opt_design
    place_design
    phys_opt_design
    route_design

    report_utilization     -file [file join $OUT vivado_util.rpt]
    report_utilization -hierarchical -file [file join $OUT vivado_util_hier.rpt]
    report_timing_summary -max_paths 10 -file [file join $OUT vivado_timing.rpt]
    report_timing -max_paths 5 -sort_by slack -file [file join $OUT vivado_crit_paths.rpt]

    set wns  [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
    set fmax [expr {1000.0 / ($PERIOD - $wns)}]
    puts [format "RESULT: WNS = %.3f ns   Fmax = %.1f MHz" $wns $fmax]
    set fh [open [file join $OUT vivado_summary.txt] w]
    puts $fh [format "part=%s N=%s period_ns=%s WNS_ns=%.3f Fmax_MHz=%.1f" $PART $N $PERIOD $wns $fmax]
    close $fh
    return
}

# ==============================================================================
# Yosys flow
# ==============================================================================
if {[llength [info commands yosys]] == 0} {
    error "synth.tcl must be run by Yosys (yosys -c) or Vivado (vivado -source)"
}

proc read_rtl {files} {
    foreach f $files { yosys read_verilog -DSYNTHESIS $f }
}

# synth_xilinx with the LUT-mapping step run through classic ABC instead of ABC9
# (ABC9's xaiger2 writer asserts on some Windows builds of Yosys; results are
# equivalent for resource estimation).
proc synth_xc7 {top {extra ""}} {
    yosys synth_xilinx -family xc7 -top $top {*}$extra -run :map_luts
    yosys opt_expr -mux_undef -noclkinv
    yosys abc -luts 2:2,3,6:5,10,20
    yosys clean
    yosys xilinx_srl -fixed -minlen 3
    yosys techmap -map +/xilinx/lut_map.v -map +/xilinx/cells_map.v -D LUT_WIDTH=6
    yosys xilinx_dffopt
    yosys opt_lut_ins -tech xilinx
    yosys synth_xilinx -family xc7 -top $top -run finalize:
}

# synth_ecp5 with classic ABC LUT4 (+PFUMX/L6MUX wide-function) mapping
proc synth_ecp5_abc {top json} {
    yosys synth_ecp5 -top $top -run :map_luts
    yosys techmap -map +/lattice/latches_map.v
    yosys abc -lut 4:7
    yosys clean
    yosys synth_ecp5 -top $top -run map_cells: -json $json
}

puts "== Yosys flow: N=$N, target ${FREQ_MHZ} MHz, reports in $OUT =="

if {$FLOW ne "ecp5"} {
    # ---------------- 1. Xilinx 7-series mapping (resource estimate) --------------
    yosys design -reset
    read_rtl $RTL
    yosys chparam -set N $N -set OUT_LANES $LANES $TOP
    synth_xc7 $TOP -flatten
    yosys tee -q -o [file join $OUT yosys_xc7_util.rpt] stat -tech xilinx

    # ---------------- 2. Component breakdown (xc7) --------------------------------
    # 2a. INT4 unpacker, combinational (the LUT overhead of sub-byte storage)
    yosys design -reset
    yosys read_verilog -DSYNTHESIS [file join $root rtl kv_unpack_dequant.v]
    yosys chparam -set PIPELINE 0 kv_unpack_dequant
    synth_xc7 kv_unpack_dequant
    yosys tee -q -o [file join $OUT yosys_xc7_unpack.rpt] stat -tech xilinx

    # 2b. KV cache controller (INT4 packed SRAMs + quantiser + pointers + read FSM)
    yosys design -reset
    yosys read_verilog -DSYNTHESIS [file join $root rtl kv_cache_controller.v]
    synth_xc7 kv_cache_controller
    yosys tee -q -o [file join $OUT yosys_xc7_kvc.rpt] stat -tech xilinx

    # 2c. One processing element
    yosys design -reset
    yosys read_verilog -DSYNTHESIS [file join $root rtl pe.v]
    synth_xc7 pe
    yosys tee -q -o [file join $OUT yosys_xc7_pe.rpt] stat -tech xilinx

}

if {$FLOW ne "xc7"} {
    # ---------------- 3. Lattice ECP5 mapping + place & route (timing) ------------
    yosys design -reset
    read_rtl $RTL
    yosys chparam -set N $N -set OUT_LANES $LANES_PNR $TOP
    synth_ecp5_abc $TOP [file join $OUT ${TOP}_ecp5.json]
    yosys tee -q -o [file join $OUT yosys_ecp5_util.rpt] stat

    if {$RUN_PNR} {
        set pnr_log [file join $OUT nextpnr_ecp5.log]
        set pnr_rpt [file join $OUT nextpnr_ecp5_report.json]
        puts "== nextpnr-ecp5 (LFE5UM-85F, CABGA756, ${FREQ_MHZ} MHz target) =="
        if {[catch {
            exec nextpnr-ecp5 --85k --package CABGA756 --speed 8 \
                --json [file join $OUT ${TOP}_ecp5.json] \
                --freq $FREQ_MHZ --timing-allow-fail --seed $SEED \
                --report $pnr_rpt --log $pnr_log 2>@1
        } msg]} {
            puts "nextpnr finished with message (see $pnr_log): [string range $msg 0 400]"
        }
        if {[file exists $pnr_log]} {
            set fh [open $pnr_log r]
            set txt [read $fh]
            close $fh
            foreach line [split $txt "\n"] {
                if {[string match "*Max frequency for clock*" $line]} { puts "RESULT: [string trim $line]" }
            }
        }
    }

}

puts "== synthesis complete; reports in $OUT =="
