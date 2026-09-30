// -----------------------------------------------------------------------------
// vcd_dump.v -- Optional VCD waveform dump for Icarus Verilog runs.
//   Compiled as an extra root module (iverilog -s vcd_dump) only when waves are
//   requested; DUMP_TOP names the design root to trace. Output: dump.vcd in the
//   simulation working directory (sim_build/<target>/).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module vcd_dump;
`ifdef DUMP_TOP
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, `DUMP_TOP);
    end
`endif
endmodule
