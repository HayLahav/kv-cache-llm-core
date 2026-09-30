// -----------------------------------------------------------------------------
// pe.v -- Output-stationary Processing Element
//
//   * Signed INT8 x INT8 multiply, INT32 accumulate.
//   * Three internal pipeline stages for high f_max:
//       S0: operand register   (DSP input register: AREG/BREG, ECP5 REG_INPUT),
//           enabled only on valid beats (operand isolation)
//       S1: product register   (DSP48 MREG / ECP5 REG_PIPELINE)
//       S2: accumulator update (DSP48 PREG / carry chain)
//   * Operand a and its control tags (valid/first/last) travel east,
//     operand b travels south; each hop is one register (systolic).
//   * first : the accumulator is re-initialised with this product
//     last  : the final sum is latched into `res` and `res_valid` pulses
//
// Synthesizable Verilog-2001. Control registers use a synchronous active-low
// reset; datapath registers are intentionally reset-free (qualified by valid).
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module pe #(
    parameter DATA_W = 8,
    parameter ACC_W  = 32
) (
    input  wire                     clk,
    input  wire                     rst_n,
    // west / north inputs
    input  wire signed [DATA_W-1:0] a_in,
    input  wire signed [DATA_W-1:0] b_in,
    input  wire                     v_in,
    input  wire                     f_in,
    input  wire                     l_in,
    // east / south outputs (registered, one-cycle systolic hop)
    output reg  signed [DATA_W-1:0] a_out,
    output reg  signed [DATA_W-1:0] b_out,
    output reg                      v_out,
    output reg                      f_out,
    output reg                      l_out,
    // result
    output reg  signed [ACC_W-1:0]  res,
    output reg                      res_valid
);

    localparam PROD_W = 2 * DATA_W;

    // ---------------- systolic forwarding ----------------
    always @(posedge clk) begin
        a_out <= a_in;
        b_out <= b_in;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            v_out <= 1'b0;
            f_out <= 1'b0;
            l_out <= 1'b0;
        end else begin
            v_out <= v_in;
            f_out <= f_in & v_in;
            l_out <= l_in & v_in;
        end
    end

    // ---------------- stage 0: operand registers ----------------
    // Operand isolation: the multiplier inputs only change on valid beats
    // (a and b of the same reduction step arrive together with v_in), which
    // cuts DSP toggling when the mesh is idle. It also keeps these flops
    // logically distinct from a_out/b_out, so synthesis cannot merge them and
    // the multiplier gets its own flops placed next to the DSP block.
    reg signed [DATA_W-1:0] a_s0, b_s0;
    reg                     v_s0, f_s0, l_s0;

    always @(posedge clk) begin
        if (v_in) begin
            a_s0 <= a_in;
            b_s0 <= b_in;
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            v_s0 <= 1'b0;
            f_s0 <= 1'b0;
            l_s0 <= 1'b0;
        end else begin
            v_s0 <= v_in;
            f_s0 <= f_in & v_in;
            l_s0 <= l_in & v_in;
        end
    end

    // ---------------- stage 1: multiply ----------------
    reg signed [PROD_W-1:0] prod_s1;
    reg                     v_s1, f_s1, l_s1;

    always @(posedge clk) begin
        prod_s1 <= a_s0 * b_s0;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            v_s1 <= 1'b0;
            f_s1 <= 1'b0;
            l_s1 <= 1'b0;
        end else begin
            v_s1 <= v_s0;
            f_s1 <= f_s0;
            l_s1 <= l_s0;
        end
    end

    // ---------------- stage 2: accumulate ----------------
    reg  signed [ACC_W-1:0] acc;
    wire signed [ACC_W-1:0] prod_ext = {{(ACC_W-PROD_W){prod_s1[PROD_W-1]}}, prod_s1};
    wire signed [ACC_W-1:0] acc_nxt  = f_s1 ? prod_ext : (acc + prod_ext);

    always @(posedge clk) begin
        if (v_s1)
            acc <= acc_nxt;
        if (v_s1 && l_s1)
            res <= acc_nxt;
    end

    always @(posedge clk) begin
        if (!rst_n)
            res_valid <= 1'b0;
        else
            res_valid <= v_s1 & l_s1;
    end

endmodule
