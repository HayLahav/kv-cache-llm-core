// -----------------------------------------------------------------------------
// kv_unpack_dequant.v -- INT4x2 packed row -> INT8 lanes
//
//   Input : one packed KV row, HEAD_DIM signed nibbles; element e occupies
//           bits [4e+3:4e] (byte b = {e(2b+1), e(2b)}).
//   Output: HEAD_DIM signed INT8 lanes, lane e at bits [8e+7:8e]:
//               out_e = sext4to8(nibble_e) <<< shift        (shift clamped to 4)
//           shift = 0 -> pure sign extension (-8..+7)
//           shift = 4 -> full INT8 range (-128..+112)
//
//   PIPELINE = 0 : purely combinational, handshake passes straight through.
//   PIPELINE = 1 : one register stage with full AXI-style valid/ready
//                  (s_ready = !m_valid || m_ready, zero-bubble throughput).
//   PIPELINE = 2 : two stages -- the packed row is registered first (so an
//                  SRAM without an output register only drives a flop), then
//                  unpacked into the output register. Same handshake rules.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module kv_unpack_dequant #(
    parameter HEAD_DIM = 16,
    parameter PIPELINE = 1
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire [2:0]            shift,
    // packed input stream
    input  wire                  s_valid,
    output wire                  s_ready,
    input  wire [HEAD_DIM*4-1:0] s_data,
    input  wire                  s_last,
    // unpacked output stream
    output wire                  m_valid,
    input  wire                  m_ready,
    output wire [HEAD_DIM*8-1:0] m_data,
    output wire                  m_last
);

    wire [2:0] sh = (shift > 3'd4) ? 3'd4 : shift;

    // ---------------- optional input register (PIPELINE == 2) ----------------
    wire                  a_valid, a_ready, a_last;
    wire [HEAD_DIM*4-1:0] a_data;

    generate
        if (PIPELINE >= 2) begin : g_in_reg
            reg                  iv_q;
            reg [HEAD_DIM*4-1:0] id_q;
            reg                  il_q;

            assign s_ready = !iv_q || a_ready;

            always @(posedge clk) begin
                if (!rst_n)
                    iv_q <= 1'b0;
                else if (s_ready)
                    iv_q <= s_valid;
            end

            always @(posedge clk) begin
                if (s_ready && s_valid) begin
                    id_q <= s_data;
                    il_q <= s_last;
                end
            end

            assign a_valid = iv_q;
            assign a_data  = id_q;
            assign a_last  = il_q;
        end else begin : g_in_pass
            assign a_valid = s_valid;
            assign a_data  = s_data;
            assign a_last  = s_last;
            assign s_ready = a_ready;
        end
    endgenerate

    // ---------------- combinational unpack lanes ----------------
    wire [HEAD_DIM*8-1:0] unpacked;

    genvar e;
    generate
        for (e = 0; e < HEAD_DIM; e = e + 1) begin : g_lane
            wire [3:0] nib = a_data[4*e +: 4];
            wire [7:0] ext = {{4{nib[3]}}, nib};      // sign-extend 4 -> 8
            assign unpacked[8*e +: 8] = ext << sh;    // dequantise by 2^sh
        end
    endgenerate

    generate
        if (PIPELINE == 0) begin : g_comb
            assign m_valid = a_valid;
            assign a_ready = m_ready;
            assign m_data  = unpacked;
            assign m_last  = a_last;
        end else begin : g_pipe
            reg                  v_q;
            reg [HEAD_DIM*8-1:0] d_q;
            reg                  l_q;

            assign a_ready = !v_q || m_ready;

            always @(posedge clk) begin
                if (!rst_n)
                    v_q <= 1'b0;
                else if (a_ready)
                    v_q <= a_valid;
            end

            always @(posedge clk) begin
                if (a_ready && a_valid) begin
                    d_q <= unpacked;
                    l_q <= a_last;
                end
            end

            assign m_valid = v_q;
            assign m_data  = d_q;
            assign m_last  = l_q;
        end
    endgenerate

endmodule
