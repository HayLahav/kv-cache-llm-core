// -----------------------------------------------------------------------------
// kv_cache_tb_top.v -- Verification harness for the KV cache subsystem
//
//   * u_kv     : kv_cache_controller under test
//   * u_unpack : pipelined kv_unpack_dequant on the controller's read stream
//                (the exact topology used inside llm_decode_top)
//   * u_raw    : a second, combinational kv_unpack_dequant with free inputs so
//                the testbench can sweep every packed nibble pattern directly
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module kv_cache_tb_top #(
    parameter HEAD_DIM    = 16,
    parameter SINK_COUNT  = 4,
    parameter WINDOW_SIZE = 64
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  clear,
    input  wire [2:0]            cfg_shift,
    // append
    input  wire                  app_valid,
    output wire                  app_ready,
    input  wire [HEAD_DIM*8-1:0] app_k,
    input  wire [HEAD_DIM*8-1:0] app_v,
    // read burst
    input  wire                  rd_start,
    input  wire                  rd_sel,
    output wire                  rd_busy,
    // raw packed stream (controller output)
    output wire                  pk_valid,
    output wire [HEAD_DIM*4-1:0] pk_data,
    output wire                  pk_phase,
    // unpacked stream
    output wire                  u_valid,
    input  wire                  u_ready,
    output wire [HEAD_DIM*8-1:0] u_data,
    output wire                  u_last,
    // status
    output wire [31:0]           token_count,
    output wire [15:0]           sink_fill,
    output wire [15:0]           win_fill,
    output wire [31:0]           head_ptr,
    // standalone combinational unpacker
    input  wire [HEAD_DIM*4-1:0] raw_packed,
    input  wire [2:0]            raw_shift,
    output wire [HEAD_DIM*8-1:0] raw_unpacked
);

    wire pk_ready, pk_last;

    kv_cache_controller #(
        .HEAD_DIM    (HEAD_DIM),
        .SINK_COUNT  (SINK_COUNT),
        .WINDOW_SIZE (WINDOW_SIZE),
        .CNT_W       (32),
        .FILL_W      (16)
    ) u_kv (
        .clk         (clk),
        .rst_n       (rst_n),
        .clear       (clear),
        .cfg_shift   (cfg_shift),
        .app_valid   (app_valid),
        .app_ready   (app_ready),
        .app_k       (app_k),
        .app_v       (app_v),
        .rd_start    (rd_start),
        .rd_sel      (rd_sel),
        .rd_busy     (rd_busy),
        .m_valid     (pk_valid),
        .m_ready     (pk_ready),
        .m_data      (pk_data),
        .m_last      (pk_last),
        .m_phase     (pk_phase),
        .token_count (token_count),
        .sink_fill   (sink_fill),
        .win_fill    (win_fill),
        .head_ptr    (head_ptr)
    );

    kv_unpack_dequant #(
        .HEAD_DIM (HEAD_DIM),
        .PIPELINE (2)
    ) u_unpack (
        .clk     (clk),
        .rst_n   (rst_n),
        .shift   (cfg_shift),
        .s_valid (pk_valid),
        .s_ready (pk_ready),
        .s_data  (pk_data),
        .s_last  (pk_last),
        .m_valid (u_valid),
        .m_ready (u_ready),
        .m_data  (u_data),
        .m_last  (u_last)
    );

    kv_unpack_dequant #(
        .HEAD_DIM (HEAD_DIM),
        .PIPELINE (0)
    ) u_raw (
        .clk     (clk),
        .rst_n   (rst_n),
        .shift   (raw_shift),
        .s_valid (1'b1),
        .s_ready (),
        .s_data  (raw_packed),
        .s_last  (1'b0),
        .m_valid (),
        .m_ready (1'b1),
        .m_data  (raw_unpacked),
        .m_last  ()
    );

endmodule
