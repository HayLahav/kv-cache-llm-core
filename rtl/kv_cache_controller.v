// -----------------------------------------------------------------------------
// kv_cache_controller.v -- Attention-Sink + Sliding-Window KV cache (INT4 packed)
//
//   Storage: two simple-dual-port SRAMs (K and V), DEPTH = SINK_COUNT + WINDOW_SIZE
//            rows, each row = HEAD_DIM signed INT4 values packed two per byte.
//
//      addr 0 .. SINK_COUNT-1                : attention sinks (written once)
//      addr SINK_COUNT .. SINK_COUNT+W-1     : circular window,
//                                              slot = SINK_COUNT + (head_ptr & (W-1))
//
//   Write path (port A): app_valid/app_ready. The INT8 K and V rows are quantised
//     to INT4 (round-half-up arithmetic shift by cfg_shift, clamp to [-8,7]) and
//     packed. While sink_fill < SINK_COUNT the row goes to the next sink slot;
//     afterwards it goes to the window and head_ptr advances. Sink addresses are
//     unreachable from the window path, so sinks are immutable after prefill.
//
//   Read path (port B): rd_start launches a two-phase burst
//       Phase A : sink slots 0 .. sink_fill-1
//       Phase B : win_fill window rows, oldest first, starting at
//                 (head_ptr - win_fill) & (W-1)
//     Rows (K or V, chosen by rd_sel) leave on a valid/ready stream with
//     m_last on the final row. The SRAM output register doubles as the stream
//     register (read enable = !m_valid || m_ready) -> full throughput, no skid.
//
//   Collision prevention: app_ready is low while a burst is in progress, so
//   port A never writes an address port B is streaming.
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module kv_cache_controller #(
    parameter HEAD_DIM    = 16,
    parameter SINK_COUNT  = 4,
    parameter WINDOW_SIZE = 64,     // must be a power of two
    parameter CNT_W       = 32,
    parameter FILL_W      = 16      // width of sink/window fill counters
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  clear,          // empty the cache (new sequence)
    input  wire [2:0]            cfg_shift,      // quantisation scale 2^shift (0..4)
    // append (token ingestion)
    input  wire                  app_valid,
    output wire                  app_ready,
    input  wire [HEAD_DIM*8-1:0] app_k,
    input  wire [HEAD_DIM*8-1:0] app_v,
    // burst read control
    input  wire                  rd_start,
    input  wire                  rd_sel,         // 0 = K, 1 = V
    output wire                  rd_busy,
    // packed row output stream
    output reg                   m_valid,
    input  wire                  m_ready,
    output wire [HEAD_DIM*4-1:0] m_data,
    output reg                   m_last,
    output reg                   m_phase,        // 0 = sink, 1 = window
    // status
    output reg  [CNT_W-1:0]      token_count,
    output reg  [FILL_W-1:0]     sink_fill,
    output reg  [FILL_W-1:0]     win_fill,
    output reg  [CNT_W-1:0]      head_ptr
);

    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            clog2 = 0;
            while (v > 0) begin
                clog2 = clog2 + 1;
                v = v >> 1;
            end
        end
    endfunction

    localparam DEPTH  = SINK_COUNT + WINDOW_SIZE;
    localparam AW     = (clog2(DEPTH) < 1) ? 1 : clog2(DEPTH);
    localparam WAW    = (clog2(WINDOW_SIZE) < 1) ? 1 : clog2(WINDOW_SIZE);
    localparam SC_W   = FILL_W;
    localparam WC_W   = FILL_W;
    localparam ROW_W  = HEAD_DIM * 4;
    localparam [CNT_W-1:0] WMASK = WINDOW_SIZE - 1;
    // burst index width: enough for max(SINK_COUNT, WINDOW_SIZE) - 1
    localparam IW     = (WINDOW_SIZE > SINK_COUNT) ? WAW : ((clog2(SINK_COUNT) < 1) ? 1 : clog2(SINK_COUNT));

    // ---------------- INT8 -> INT4 quantiser ----------------
    // round-half-up:  floor((x + 2^(s-1)) / 2^s) == (x >>> s) + x[s-1]
    // so no 9-bit adder is needed: arithmetic shift, pick the round bit, then
    // saturate (detected on the shifted value) or do a 4-bit increment.
    function [3:0] quant4;
        input [7:0] x;
        input [2:0] s;
        reg   [2:0]        sc;
        reg   signed [7:0] sh;
        reg                rb, pos_ovf, neg_ovf, at_max;
        begin
            sc = (s > 3'd4) ? 3'd4 : s;
            sh = $signed(x) >>> sc;
            rb = (sc == 3'd0) ? 1'b0 : x[sc - 3'd1];
            // range checks as plain bit logic (no carry-chain comparators):
            // sh fits INT4 iff bits [7:3] are all equal to the sign bit
            pos_ovf = ~sh[7] &  (|sh[6:3]);                 // sh >  7
            neg_ovf =  sh[7] & ~(&sh[6:3]);                 // sh < -8
            at_max  = ~sh[7] & ~(|sh[6:3]) & (&sh[2:0]);    // sh == 7
            if (pos_ovf || (at_max && rb))
                quant4 = 4'd7;
            else if (neg_ovf)
                quant4 = 4'b1000;
            else
                quant4 = sh[3:0] + {3'b000, rb};
        end
    endfunction

    // Append pipeline (off the critical path: one APPEND takes >= 3 AXI beats).
    // A single HEAD_DIM-lane quantiser is time-shared between K and V, halving
    // its LUT cost; accepts are therefore spaced >= 2 cycles apart. The
    // quantiser is fed from a dedicated input register (q_x / q_s), so the K/V
    // selection happens *before* a flop and never sits in front of the
    // quantiser logic.
    //   W0 accept : pointers advance; q_x <= K row, V row parked
    //   W1        : quantise + pack K;    q_x <= V row
    //   W2        : quantise + pack V
    //   W3        : K and V SRAM write
    reg                  w1_valid, w2_valid, w3_valid;
    reg [HEAD_DIM*8-1:0] q_x;            // quantiser input register
    reg [2:0]            q_s;
    reg [HEAD_DIM*8-1:0] v_park;
    reg [AW-1:0]         w1_addr, w2_addr, w3_addr;
    reg [ROW_W-1:0]      w2_k, w3_k, w3_v;
    wire                 wr_inflight = w1_valid | w2_valid | w3_valid;

    wire [ROW_W-1:0] q_out;
    genvar e;
    generate
        for (e = 0; e < HEAD_DIM; e = e + 1) begin : g_pack
            assign q_out[4*e +: 4] = quant4(q_x[8*e +: 8], q_s);
        end
    endgenerate

    // ---------------- read FSM state ----------------
    localparam [1:0] RD_IDLE = 2'd0,
                     RD_SINK = 2'd1,
                     RD_WIN  = 2'd2;

    reg [1:0]       rd_state;
    reg             sel_q;
    reg [IW-1:0]    idx;
    reg [IW-1:0]    sink_last;       // sink_fill - 1, registered at burst start
    reg [IW-1:0]    win_last;        // win_fill  - 1
    reg             has_win;
    reg [WAW-1:0]   win_start;       // oldest window slot (relative)
    reg             start_pending;   // rd_start seen while a write was in flight
    reg             sel_pending;

    wire start_req = rd_start | start_pending;
    wire start_go  = (rd_state == RD_IDLE) && start_req && !wr_inflight;

    assign app_ready = (rd_state == RD_IDLE) && !start_req && !clear && !w1_valid;
    assign rd_busy   = (rd_state != RD_IDLE) || m_valid || start_pending;

    // ---------------- write port (append) ----------------
    wire             do_app  = app_valid && app_ready;
    wire             to_sink = (sink_fill < SINK_COUNT);
    wire [CNT_W-1:0] win_slot = SINK_COUNT + (head_ptr & WMASK);
    wire [AW-1:0]    waddr    = to_sink ? sink_fill : win_slot[AW-1:0];

    reg [ROW_W-1:0] kmem [0:DEPTH-1];
    reg [ROW_W-1:0] vmem [0:DEPTH-1];

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            w1_valid <= 1'b0;
            w2_valid <= 1'b0;
            w3_valid <= 1'b0;
        end else begin
            w1_valid <= do_app;
            w2_valid <= w1_valid;
            w3_valid <= w2_valid;
        end
    end

    // do_app and w1_valid are mutually exclusive (app_ready requires !w1_valid)
    always @(posedge clk) begin
        if (do_app) begin
            q_x     <= app_k;
            q_s     <= cfg_shift;
            v_park  <= app_v;
            w1_addr <= waddr;
        end else if (w1_valid) begin
            q_x     <= v_park;
        end
        if (w1_valid) begin
            w2_k    <= q_out;           // packed K
            w2_addr <= w1_addr;
        end
        if (w2_valid) begin
            w3_k    <= w2_k;
            w3_v    <= q_out;           // packed V
            w3_addr <= w2_addr;
        end
        if (w3_valid) begin
            kmem[w3_addr] <= w3_k;
            vmem[w3_addr] <= w3_v;
        end
    end

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            token_count <= {CNT_W{1'b0}};
            sink_fill   <= {SC_W{1'b0}};
            win_fill    <= {WC_W{1'b0}};
            head_ptr    <= {CNT_W{1'b0}};
        end else if (do_app) begin
            if (token_count != {CNT_W{1'b1}})
                token_count <= token_count + 1'b1;
            if (to_sink) begin
                sink_fill <= sink_fill + 1'b1;
            end else begin
                head_ptr <= head_ptr + 1'b1;
                if (win_fill != WINDOW_SIZE)
                    win_fill <= win_fill + 1'b1;
            end
        end
    end

    // ---------------- read port (two-phase burst) ----------------
    wire rd_en = (rd_state != RD_IDLE) && (!m_valid || m_ready);

    // WAW-bit addition wraps modulo WINDOW_SIZE for free (power of two)
    wire [WAW-1:0]   win_rel  = win_start + idx[WAW-1:0];
    wire [AW-1:0]    win_addr = SINK_COUNT + win_rel;
    wire [AW-1:0]    raddr    = (rd_state == RD_SINK) ? idx : win_addr;

    // end-of-phase detection is a pure equality against registered bounds
    wire sink_done = (rd_state == RD_SINK) && (idx == sink_last);
    wire win_done  = (rd_state == RD_WIN)  && (idx == win_last);
    wire is_last   = win_done || (sink_done && !has_win);

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            rd_state  <= RD_IDLE;
            sel_q     <= 1'b0;
            idx       <= {IW{1'b0}};
            sink_last <= {IW{1'b0}};
            win_last  <= {IW{1'b0}};
            has_win   <= 1'b0;
            win_start <= {WAW{1'b0}};
            start_pending <= 1'b0;
            sel_pending   <= 1'b0;
        end else begin
            // a burst requested while the append pipeline drains is deferred
            if (rd_start && !start_go) begin
                start_pending <= 1'b1;
                sel_pending   <= rd_sel;
            end else if (start_go) begin
                start_pending <= 1'b0;
            end
            case (rd_state)
                RD_IDLE: begin
                    if (start_go) begin
                        sel_q     <= start_pending ? sel_pending : rd_sel;
                        idx       <= {IW{1'b0}};
                        sink_last <= sink_fill - 1'b1;
                        win_last  <= win_fill  - 1'b1;
                        has_win   <= (win_fill != 0);
                        win_start <= head_ptr[WAW-1:0] - win_fill[WAW-1:0];
                        if (sink_fill != 0)
                            rd_state <= RD_SINK;
                        else if (win_fill != 0)
                            rd_state <= RD_WIN;
                    end
                end
                RD_SINK: begin
                    if (rd_en) begin
                        if (sink_done) begin
                            idx      <= {IW{1'b0}};
                            rd_state <= has_win ? RD_WIN : RD_IDLE;
                        end else begin
                            idx <= idx + 1'b1;
                        end
                    end
                end
                RD_WIN: begin
                    if (rd_en) begin
                        if (win_done) begin
                            idx      <= {IW{1'b0}};
                            rd_state <= RD_IDLE;
                        end else begin
                            idx <= idx + 1'b1;
                        end
                    end
                end
                default: rd_state <= RD_IDLE;
            endcase
        end
    end

    // SRAM output registers act as the stream register
    reg [ROW_W-1:0] kq, vq;
    always @(posedge clk) begin
        if (rd_en && !sel_q) kq <= kmem[raddr];   // only the selected SRAM toggles
        if (rd_en &&  sel_q) vq <= vmem[raddr];
    end

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            m_valid <= 1'b0;
            m_last  <= 1'b0;
            m_phase <= 1'b0;
        end else if (rd_en) begin
            m_valid <= 1'b1;
            m_last  <= is_last;
            m_phase <= (rd_state == RD_WIN);
        end else if (m_ready) begin
            m_valid <= 1'b0;
        end
    end

    assign m_data = sel_q ? vq : kq;

endmodule
