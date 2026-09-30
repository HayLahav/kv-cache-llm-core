// -----------------------------------------------------------------------------
// ping_pong_buffer.v -- Dual-bank operand buffer with bank-switching FSM
//
//   Two independent SRAM banks (each DEPTH x WIDTH, 1 write + 1 sync-read port).
//   The producer (prefetch from Flash/NVM via AXI) always fills bank `wr_bank`
//   while the consumer (systolic array feed) reads bank `rd_bank`.
//
//   Per-bank state machine:
//
//        wr_en            wr_commit             rd_acquire           rd_release
//   EMPTY ─────► FILLING ───────────► READY ───────────────► COMPUTE ──────────► EMPTY
//     └───────────── wr_commit ─────────┘
//
//   * wr_ready = state[wr_bank] in {EMPTY, FILLING}   (producer may write)
//   * rd_ready = state[rd_bank] == READY              (consumer may acquire)
//   * wr_commit toggles wr_bank, rd_release toggles rd_bank.
//
//   Collision-freedom: a bank is written only while EMPTY/FILLING and read only
//   while READY/COMPUTE, so the two ports never touch the same bank at once.
//   The read port is synchronous for block-RAM inference; rd_data holds its
//   value when no read is in flight.
//     OUT_REG = 0 : 1-cycle read latency
//     OUT_REG = 1 : 2-cycle read latency; each bank's output is re-registered
//                   before the bank-select mux (maps onto the BRAM output
//                   register, removing the slow clock-to-out from the path)
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module ping_pong_buffer #(
    parameter WIDTH = 32,
    parameter DEPTH = 128,
    parameter AW    = 7,
    parameter OUT_REG = 1
) (
    input  wire             clk,
    input  wire             rst_n,
    // producer (prefetch) port
    input  wire             wr_en,
    input  wire [AW-1:0]    wr_addr,
    input  wire [WIDTH-1:0] wr_data,
    input  wire             wr_commit,
    output wire             wr_ready,
    // consumer (compute) port
    input  wire             rd_acquire,
    input  wire             rd_release,
    input  wire             rd_en,
    input  wire [AW-1:0]    rd_addr,
    output wire [WIDTH-1:0] rd_data,
    output wire             rd_ready,
    // status
    output reg              wr_bank,
    output reg              rd_bank,
    output wire [3:0]       bank_state      // {state[1], state[0]}
);

    localparam [1:0] S_EMPTY   = 2'd0,
                     S_FILLING = 2'd1,
                     S_READY   = 2'd2,
                     S_COMPUTE = 2'd3;

    reg [1:0] state0, state1;

    wire [1:0] wr_state = wr_bank ? state1 : state0;
    wire [1:0] rd_state = rd_bank ? state1 : state0;

    assign wr_ready   = (wr_state == S_EMPTY) || (wr_state == S_FILLING);
    assign rd_ready   = (rd_state == S_READY);
    assign bank_state = {state1, state0};

    wire do_wr      = wr_en     & wr_ready;
    wire do_commit  = wr_commit & wr_ready;
    wire do_acquire = rd_acquire & rd_ready;
    wire do_release = rd_release & (rd_state == S_COMPUTE);

    // next-state for one bank
    function [1:0] next_state;
        input [1:0] cur;
        input       is_wr_bank;
        input       is_rd_bank;
        input       f_wr;
        input       f_commit;
        input       f_acquire;
        input       f_release;
        begin
            next_state = cur;
            case (cur)
                S_EMPTY:   if (is_wr_bank && f_commit)        next_state = S_READY;
                           else if (is_wr_bank && f_wr)      next_state = S_FILLING;
                S_FILLING: if (is_wr_bank && f_commit)        next_state = S_READY;
                S_READY:   if (is_rd_bank && f_acquire)       next_state = S_COMPUTE;
                S_COMPUTE: if (is_rd_bank && f_release)       next_state = S_EMPTY;
                default:                                      next_state = S_EMPTY;
            endcase
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            state0  <= S_EMPTY;
            state1  <= S_EMPTY;
            wr_bank <= 1'b0;
            rd_bank <= 1'b0;
        end else begin
            state0 <= next_state(state0, wr_bank == 1'b0, rd_bank == 1'b0,
                                 do_wr, do_commit, do_acquire, do_release);
            state1 <= next_state(state1, wr_bank == 1'b1, rd_bank == 1'b1,
                                 do_wr, do_commit, do_acquire, do_release);
            if (do_commit)  wr_bank <= ~wr_bank;
            if (do_release) rd_bank <= ~rd_bank;
        end
    end

    // ---------------- storage: two simple-dual-port banks ----------------
    reg [WIDTH-1:0] mem0 [0:DEPTH-1];
    reg [WIDTH-1:0] mem1 [0:DEPTH-1];
    reg [WIDTH-1:0] q0, q1;
    reg             rd_sel_q;

    always @(posedge clk) begin
        if (do_wr && !wr_bank) mem0[wr_addr] <= wr_data;
        if (rd_en)             q0 <= mem0[rd_addr];
    end

    always @(posedge clk) begin
        if (do_wr && wr_bank)  mem1[wr_addr] <= wr_data;
        if (rd_en)             q1 <= mem1[rd_addr];
    end

    always @(posedge clk) begin
        if (!rst_n)     rd_sel_q <= 1'b0;
        else if (rd_en) rd_sel_q <= rd_bank;
    end

    generate
        if (OUT_REG) begin : g_oreg
            reg             rd_en_q;
            reg [WIDTH-1:0] q0_r, q1_r;
            reg             rd_sel_r;
            always @(posedge clk) begin
                if (!rst_n) rd_en_q <= 1'b0;
                else        rd_en_q <= rd_en;
            end
            always @(posedge clk) begin
                if (rd_en_q) begin
                    q0_r     <= q0;
                    q1_r     <= q1;
                    rd_sel_r <= rd_sel_q;
                end
            end
            assign rd_data = rd_sel_r ? q1_r : q0_r;
        end else begin : g_noreg
            assign rd_data = rd_sel_q ? q1 : q0;
        end
    endgenerate

endmodule
