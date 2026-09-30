// -----------------------------------------------------------------------------
// llm_decode_top.v -- Edge LLM Decode Core (AXI4-Stream wrapped)
//
//   s_axis (HEAD_DIM*8 bits) --> Command Parser
//        CFG     0x0 : hdr[10:8] = INT4 scale shift (0..4)
//        APPEND  0x1 : +K row beat, +V row beat (tlast)   -> KV cache (INT4 packed)
//        LOAD    0x2 : +1..BUF_DEPTH words (low N*8 bits, tlast on last)
//                                                         -> ping-pong prefetch bank
//        SCORE   0x3 : S[tok][h] = sum_d K[tok][d] * Q[h][d]   (bank word d = Q^T col)
//        CONTEXT 0x4 : O[h][c]   = sum_t P[h][t] * V[t][c]     (bank word t = P^T col)
//        CLEAR   0x5 : empty the KV cache
//   m_axis (OUT_LANES*32 bits) <-- signed INT32 results, OUT_LANES consecutive
//        words per beat (lane k at bits [32k+31:32k]); tlast on the last beat of
//        an op. OUT_LANES = N (default) emits one whole array row per beat.
//
//   SCORE / CONTEXT are dispatched to the compute engine and the parser moves
//   on immediately, so a LOAD for the next step streams into the idle bank
//   while the array computes (ping-pong prefetch). APPEND / CFG / CLEAR wait
//   for the engine to go idle (KV SRAM read/write collision prevention).
//
//   Datapath:  KV SRAM -> unpack/dequant -> (tile former | V lane select)
//              -> feed stage (+ bank read) -> systolic array -> capture -> m_axis
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module llm_decode_top #(
    parameter N           = 4,
    parameter HEAD_DIM    = 16,
    parameter SINK_COUNT  = 4,
    parameter WINDOW_SIZE = 64,
    parameter BUF_DEPTH   = 128,
    parameter BUF_AW      = 7,      // clog2(BUF_DEPTH)
    parameter PP_OUT_REG  = 1,      // ping-pong read latency 1 (0) or 2 (1) cycles
    parameter OUT_LANES   = N       // INT32 results per m_axis beat; must divide N
) (
    input  wire                  clk,
    input  wire                  rst_n,
    // AXI4-Stream slave (commands + data)
    input  wire [HEAD_DIM*8-1:0] s_axis_tdata,
    input  wire                  s_axis_tvalid,
    output wire                  s_axis_tready,
    input  wire                  s_axis_tlast,
    // AXI4-Stream master (INT32 results)
    output wire [OUT_LANES*32-1:0] m_axis_tdata,
    output wire                  m_axis_tvalid,
    input  wire                  m_axis_tready,
    output wire                  m_axis_tlast,
    // status
    output wire                  busy,
    output wire [31:0]           token_count
);

    // ============================ constants ===============================
    localparam IN_W    = HEAD_DIM * 8;
    localparam ROW_W   = HEAD_DIM * 4;
    localparam LANE_W  = N * 8;
    localparam ACC_W   = 32;
    localparam PASSES  = HEAD_DIM / N;
    // Minimum spacing of overlapped 'last' beats. The previous tile completes
    // 2(N-1)+4 cycles after its last beat and PE(0,0) latches the next tile 4
    // cycles after that tile's last beat, so a spacing >= 2N-1 is sufficient;
    // 2N+2 leaves margin for the capture handshake.
    localparam GAP     = 2 * N + 2;

    localparam [3:0] OP_CFG     = 4'h0,
                     OP_APPEND  = 4'h1,
                     OP_LOAD    = 4'h2,
                     OP_SCORE   = 4'h3,
                     OP_CONTEXT = 4'h4,
                     OP_CLEAR   = 4'h5;

    // ============================ signals =================================
    // parser
    localparam [2:0] P_HDR   = 3'd0,
                     P_APP_K = 3'd1,
                     P_APP_V = 3'd2,
                     P_LOAD  = 3'd3,
                     P_WAIT  = 3'd4;
    reg  [2:0]        p_state;
    reg  [3:0]        p_op;
    reg  [2:0]        p_arg;
    reg  [IN_W-1:0]   app_k_q;
    reg  [BUF_AW-1:0] ld_addr;
    reg  [2:0]        cfg_shift;

    // engine
    localparam [2:0] E_IDLE  = 3'd0,
                     E_WBUF  = 3'd1,
                     E_RUN   = 3'd2,
                     E_NEXT  = 3'd3,
                     E_DRAIN = 3'd4;
    reg  [2:0]        e_state;
    reg               e_mode;          // 0 = SCORE, 1 = CONTEXT
    reg  [7:0]        e_pass;
    wire              eng_idle = (e_state == E_IDLE);
    reg               eng_start;
    reg               eng_start_mode;

    // KV controller
    wire              kv_app_valid, kv_app_ready;
    reg               kv_clear;
    reg               kv_rd_start;
    wire              kv_rd_busy;
    wire              kv_m_valid, kv_m_ready, kv_m_last, kv_m_phase;
    wire [ROW_W-1:0]  kv_m_data;
    wire [15:0]       kv_sink_fill, kv_win_fill;
    wire [31:0]       kv_head_ptr;

    // unpacker
    wire              u_valid, u_ready, u_last;
    wire [IN_W-1:0]   u_data;

    // ping-pong buffer
    wire              pp_wr_en, pp_wr_commit, pp_wr_ready;
    wire              pp_rd_ready;
    reg               pp_rd_acquire, pp_rd_release;
    wire              pp_rd_en;
    wire [BUF_AW-1:0] pp_rd_addr;
    wire [LANE_W-1:0] pp_rd_data;
    wire              pp_wr_bank, pp_rd_bank;
    wire [3:0]        pp_bank_state;

    // feed / array
    reg               f_valid, f_first, f_last, f_mode;
    reg  [LANE_W-1:0] f_op;
    // array-side copies, aligned with the ping-pong read latency
    wire              a_valid, a_first, a_last, a_mode;
    wire [LANE_W-1:0] a_op;
    wire [LANE_W-1:0] arr_a = a_mode ? pp_rd_data : a_op;
    wire [LANE_W-1:0] arr_b = a_mode ? a_op       : pp_rd_data;
    wire [N*N*ACC_W-1:0] arr_c;
    wire              arr_done;

    // tile bookkeeping
    reg  [1:0]        pending;         // tiles fully fed but not yet captured
    reg  [7:0]        gap_cnt;         // cycles since the last 'last' beat
    reg               gap_ok;          // registered (gap_cnt >= GAP)
    reg               hold;            // array finished while capture buffer full

    // ======================= AXI4-Stream slave / parser ===================
    wire s_hs = s_axis_tvalid & s_axis_tready;

    assign s_axis_tready = (p_state == P_HDR)   ? 1'b1 :
                           (p_state == P_APP_K) ? 1'b1 :
                           (p_state == P_APP_V) ? (eng_idle & !eng_start & kv_app_ready) :
                           (p_state == P_LOAD)  ? pp_wr_ready :
                                                  1'b0;

    assign kv_app_valid = (p_state == P_APP_V) & s_axis_tvalid & eng_idle & !eng_start;
    assign pp_wr_en     = (p_state == P_LOAD) & s_hs;
    assign pp_wr_commit = (p_state == P_LOAD) & s_hs & s_axis_tlast;

    always @(posedge clk) begin
        if (!rst_n) begin
            p_state        <= P_HDR;
            p_op           <= 4'd0;
            p_arg          <= 3'd0;
            ld_addr        <= {BUF_AW{1'b0}};
            cfg_shift      <= 3'd0;
            kv_clear       <= 1'b0;
            eng_start      <= 1'b0;
            eng_start_mode <= 1'b0;
        end else begin
            kv_clear  <= 1'b0;
            eng_start <= 1'b0;
            case (p_state)
                P_HDR: begin
                    if (s_hs) begin
                        p_op  <= s_axis_tdata[3:0];
                        p_arg <= s_axis_tdata[10:8];
                        case (s_axis_tdata[3:0])
                            OP_APPEND: p_state <= P_APP_K;
                            OP_LOAD: begin
                                ld_addr <= {BUF_AW{1'b0}};
                                p_state <= P_LOAD;
                            end
                            OP_CFG, OP_SCORE, OP_CONTEXT, OP_CLEAR:
                                p_state <= P_WAIT;
                            default: p_state <= P_HDR;   // unknown opcode: dropped
                        endcase
                    end
                end
                P_APP_K: begin
                    if (s_hs) p_state <= P_APP_V;
                end
                P_APP_V: begin
                    if (s_hs) p_state <= P_HDR;
                end
                P_LOAD: begin
                    if (s_hs) begin
                        ld_addr <= ld_addr + 1'b1;
                        if (s_axis_tlast) p_state <= P_HDR;
                    end
                end
                P_WAIT: begin
                    // eng_start is checked so a just-dispatched op counts as busy
                    if (eng_idle && !eng_start) begin
                        case (p_op)
                            OP_CFG:     cfg_shift <= (p_arg > 3'd4) ? 3'd4 : p_arg;
                            OP_CLEAR:   kv_clear  <= 1'b1;
                            OP_SCORE: begin
                                eng_start      <= 1'b1;
                                eng_start_mode <= 1'b0;
                            end
                            OP_CONTEXT: begin
                                eng_start      <= 1'b1;
                                eng_start_mode <= 1'b1;
                            end
                            default: ;
                        endcase
                        p_state <= P_HDR;
                    end
                end
                default: p_state <= P_HDR;
            endcase
        end
    end

    always @(posedge clk) begin
        if (p_state == P_APP_K && s_hs)
            app_k_q <= s_axis_tdata;
    end

    // ============================ KV cache ================================
    kv_cache_controller #(
        .HEAD_DIM    (HEAD_DIM),
        .SINK_COUNT  (SINK_COUNT),
        .WINDOW_SIZE (WINDOW_SIZE),
        .CNT_W       (32),
        .FILL_W      (16)
    ) u_kv (
        .clk         (clk),
        .rst_n       (rst_n),
        .clear       (kv_clear),
        .cfg_shift   (cfg_shift),
        .app_valid   (kv_app_valid),
        .app_ready   (kv_app_ready),
        .app_k       (app_k_q),
        .app_v       (s_axis_tdata),
        .rd_start    (kv_rd_start),
        .rd_sel      (e_mode),
        .rd_busy     (kv_rd_busy),
        .m_valid     (kv_m_valid),
        .m_ready     (kv_m_ready),
        .m_data      (kv_m_data),
        .m_last      (kv_m_last),
        .m_phase     (kv_m_phase),
        .token_count (token_count),
        .sink_fill   (kv_sink_fill),
        .win_fill    (kv_win_fill),
        .head_ptr    (kv_head_ptr)
    );

    kv_unpack_dequant #(
        .HEAD_DIM (HEAD_DIM),
        .PIPELINE (2)               // register the SRAM output before unpacking
    ) u_unpack (
        .clk     (clk),
        .rst_n   (rst_n),
        .shift   (cfg_shift),
        .s_valid (kv_m_valid),
        .s_ready (kv_m_ready),
        .s_data  (kv_m_data),
        .s_last  (kv_m_last),
        .m_valid (u_valid),
        .m_ready (u_ready),
        .m_data  (u_data),
        .m_last  (u_last)
    );

    // ========================= ping-pong buffer ===========================
    ping_pong_buffer #(
        .WIDTH (LANE_W),
        .DEPTH   (BUF_DEPTH),
        .AW      (BUF_AW),
        .OUT_REG (PP_OUT_REG)
    ) u_pp (
        .clk        (clk),
        .rst_n      (rst_n),
        .wr_en      (pp_wr_en),
        .wr_addr    (ld_addr),
        .wr_data    (s_axis_tdata[LANE_W-1:0]),
        .wr_commit  (pp_wr_commit),
        .wr_ready   (pp_wr_ready),
        .rd_acquire (pp_rd_acquire),
        .rd_release (pp_rd_release),
        .rd_en      (pp_rd_en),
        .rd_addr    (pp_rd_addr),
        .rd_data    (pp_rd_data),
        .rd_ready   (pp_rd_ready),
        .wr_bank    (pp_wr_bank),
        .rd_bank    (pp_rd_bank),
        .bank_state (pp_bank_state)
    );

    // ======================= tile issue permission ========================
    // A tile may start while one earlier tile is still in flight only if the
    // capture buffer is free (that earlier tile is then captured the instant
    // it completes).  Its 'last' beat must trail the previous 'last' by at
    // least GAP cycles so no PE overwrites a result before it is captured.
    wire cap_full;
    wire first_ok = (pending == 2'd0) || ((pending == 2'd1) && !cap_full);
    wire last_ok  = (pending == 2'd0) || gap_ok;

    // ============================ SCORE path ==============================
    // Tile former: N unpacked K rows -> tile registers, then HEAD_DIM beats.
    reg  [IN_W-1:0]   tile_row [0:N-1];
    reg  [7:0]        tile_cnt;        // rows loaded
    reg  [7:0]        tile_rows;       // valid rows in the full tile
    reg               tile_full;
    reg               tile_is_last;
    reg  [7:0]        sc_d;            // reduction index d within the tile

    wire sc_run   = (e_state == E_RUN) && !e_mode;
    wire sc_first = (sc_d == 8'd0);
    wire sc_last  = (sc_d == HEAD_DIM - 1);
    wire sc_go    = sc_run && tile_full &&
                    (!sc_first || first_ok) && (!sc_last || last_ok);

    // ============================ CONTEXT path ============================
    reg  [7:0]        cx_t;            // token index within the pass

    wire cx_run   = (e_state == E_RUN) && e_mode;
    wire cx_first = (cx_t == 8'd0);
    wire cx_ok    = (!cx_first || first_ok) && (!u_last || last_ok);
    wire cx_go    = cx_run && u_valid && cx_ok;

    assign u_ready = sc_run ? (!tile_full) :
                     cx_run ? cx_ok        :
                              1'b0;

    wire sc_load = sc_run && u_valid && !tile_full;

    // ============================ feed stage ==============================
    wire feed_go   = sc_go | cx_go;
    wire feed_last = sc_go ? sc_last : (cx_go & u_last);
    wire feed_first= sc_go ? sc_first : (cx_go & cx_first);

    assign pp_rd_en   = feed_go;
    assign pp_rd_addr = sc_go ? sc_d[BUF_AW-1:0] : cx_t[BUF_AW-1:0];

    // Operand registered alongside the synchronous bank read
    reg [LANE_W-1:0] sc_col;
    integer r;
    always @(*) begin
        for (r = 0; r < N; r = r + 1)
            sc_col[r*8 +: 8] = (r < tile_rows) ? tile_row[r][sc_d*8 +: 8] : 8'd0;
    end
    wire [LANE_W-1:0] cx_lanes = u_data[e_pass*LANE_W +: LANE_W];

    always @(posedge clk) begin
        if (!rst_n) begin
            f_valid <= 1'b0;
            f_first <= 1'b0;
            f_last  <= 1'b0;
            f_mode  <= 1'b0;
        end else begin
            f_valid <= feed_go;
            f_first <= feed_go & feed_first;
            f_last  <= feed_go & feed_last;
            if (feed_go) f_mode <= e_mode;
        end
    end

    always @(posedge clk) begin
        if (feed_go)
            f_op <= sc_go ? sc_col : cx_lanes;
    end

    // With a 2-cycle bank read, the operand and tags wait one more cycle.
    generate
        if (PP_OUT_REG) begin : g_feed2
            reg              f2_valid, f2_first, f2_last, f2_mode;
            reg [LANE_W-1:0] f2_op;
            always @(posedge clk) begin
                if (!rst_n) begin
                    f2_valid <= 1'b0;
                    f2_first <= 1'b0;
                    f2_last  <= 1'b0;
                    f2_mode  <= 1'b0;
                end else begin
                    f2_valid <= f_valid;
                    f2_first <= f_first;
                    f2_last  <= f_last;
                    f2_mode  <= f_mode;
                end
            end
            always @(posedge clk) begin
                if (f_valid) f2_op <= f_op;
            end
            assign a_valid = f2_valid;
            assign a_first = f2_first;
            assign a_last  = f2_last;
            assign a_mode  = f2_mode;
            assign a_op    = f2_op;
        end else begin : g_feed1
            assign a_valid = f_valid;
            assign a_first = f_first;
            assign a_last  = f_last;
            assign a_mode  = f_mode;
            assign a_op    = f_op;
        end
    endgenerate

    // ======================== result metadata FIFO ========================
    // One entry per tile, pushed when its last beat is fed, popped on capture.
    reg  [7:0]  meta_rows [0:1];
    reg         meta_end  [0:1];
    reg         meta_wp, meta_rp;

    wire meta_push     = feed_go & feed_last;
    wire meta_push_end = sc_go ? tile_is_last : (e_pass == PASSES - 1);
    wire [7:0] meta_push_rows = sc_go ? tile_rows : N;

    // ========================= systolic array =============================
    systolic_array #(
        .N      (N),
        .DATA_W (8),
        .ACC_W  (ACC_W)
    ) u_array (
        .clk       (clk),
        .rst_n     (rst_n),
        .a_in      (arr_a),
        .b_in      (arr_b),
        .in_valid  (a_valid),
        .in_first  (a_first),
        .in_last   (a_last),
        .c_flat    (arr_c),
        .out_valid (arr_done)
    );

    // ======================= capture + serializer =========================
    reg  [N*N*ACC_W-1:0] cap_data;
    reg                  cap_full_q;
    reg  [7:0]           cap_rows;
    reg                  cap_end;
    reg  [7:0]           o_i, o_j;

    assign cap_full = cap_full_q;

    wire cap_do  = (arr_done | hold) & !cap_full_q;
    wire m_hs    = m_axis_tvalid & m_axis_tready;
    wire o_row_end = (o_j == N - OUT_LANES);
    wire o_lastw   = (o_i == cap_rows - 1) && o_row_end;

    assign m_axis_tvalid = cap_full_q;
    // row i of the tile is contiguous in cap_data, so OUT_LANES consecutive
    // results C[i][j .. j+OUT_LANES-1] are a single part-select
    assign m_axis_tdata  = cap_data[(o_i*N + o_j)*ACC_W +: OUT_LANES*ACC_W];
    assign m_axis_tlast  = cap_end & o_lastw;

    always @(posedge clk) begin
        if (!rst_n) begin
            cap_full_q <= 1'b0;
            cap_rows   <= 8'd0;     // reset so m_axis_tlast is never X
            cap_end    <= 1'b0;
            hold       <= 1'b0;
            o_i        <= 8'd0;
            o_j        <= 8'd0;
            meta_wp    <= 1'b0;
            meta_rp    <= 1'b0;
            pending    <= 2'd0;
            gap_cnt    <= 8'd0;
            gap_ok     <= 1'b1;
        end else begin
            // metadata push
            if (meta_push) begin
                meta_rows[meta_wp] <= meta_push_rows;
                meta_end[meta_wp]  <= meta_push_end;
                meta_wp            <= ~meta_wp;
            end

            // pending tile counter
            case ({meta_push, cap_do})
                2'b10:   pending <= pending + 1'b1;
                2'b01:   pending <= pending - 1'b1;
                default: pending <= pending;
            endcase

            // gap counter between 'last' beats; gap_ok rises exactly GAP
            // cycles after the previous last beat (no comparator on the path)
            if (meta_push) begin
                gap_cnt <= 8'd0;
                gap_ok  <= 1'b0;
            end else if (!gap_ok) begin
                gap_cnt <= gap_cnt + 1'b1;
                if (gap_cnt == GAP - 1)
                    gap_ok <= 1'b1;
            end

            // array completion while the capture buffer is busy
            if (cap_do)
                hold <= 1'b0;
            else if (arr_done)
                hold <= 1'b1;

            // capture
            if (cap_do) begin
                cap_data   <= arr_c;
                cap_rows   <= meta_rows[meta_rp];
                cap_end    <= meta_end[meta_rp];
                meta_rp    <= ~meta_rp;
                cap_full_q <= 1'b1;
                o_i        <= 8'd0;
                o_j        <= 8'd0;
            end else if (m_hs) begin
                if (o_lastw) begin
                    cap_full_q <= 1'b0;
                end else if (o_row_end) begin
                    o_j <= 8'd0;
                    o_i <= o_i + 1'b1;
                end else begin
                    o_j <= o_j + OUT_LANES;
                end
            end
        end
    end

    // ============================ engine FSM ==============================
    always @(posedge clk) begin
        if (!rst_n) begin
            e_state       <= E_IDLE;
            e_mode        <= 1'b0;
            e_pass        <= 8'd0;
            kv_rd_start   <= 1'b0;
            pp_rd_acquire <= 1'b0;
            pp_rd_release <= 1'b0;
            tile_cnt      <= 8'd0;
            tile_rows     <= 8'd0;
            tile_full     <= 1'b0;
            tile_is_last  <= 1'b0;
            sc_d          <= 8'd0;
            cx_t          <= 8'd0;
        end else begin
            kv_rd_start   <= 1'b0;
            pp_rd_acquire <= 1'b0;
            pp_rd_release <= 1'b0;

            case (e_state)
                E_IDLE: begin
                    if (eng_start) begin
                        e_mode <= eng_start_mode;
                        // an op on an empty cache is a no-op
                        if (token_count != 32'd0)
                            e_state <= E_WBUF;
                    end
                end

                E_WBUF: begin
                    if (pp_rd_ready) begin
                        pp_rd_acquire <= 1'b1;
                        kv_rd_start   <= 1'b1;
                        e_pass        <= 8'd0;
                        tile_cnt      <= 8'd0;
                        tile_full     <= 1'b0;
                        sc_d          <= 8'd0;
                        cx_t          <= 8'd0;
                        e_state       <= E_RUN;
                    end
                end

                E_RUN: begin
                    if (!e_mode) begin
                        // ---- SCORE: tile former ----
                        if (sc_load) begin
                            tile_row[tile_cnt] <= u_data;
                            if (u_last || tile_cnt == N - 1) begin
                                tile_full    <= 1'b1;
                                tile_rows    <= tile_cnt + 1'b1;
                                tile_is_last <= u_last;
                                tile_cnt     <= 8'd0;
                            end else begin
                                tile_cnt <= tile_cnt + 1'b1;
                            end
                        end
                        // ---- SCORE: stream tile columns ----
                        if (sc_go) begin
                            if (sc_last) begin
                                sc_d      <= 8'd0;
                                tile_full <= 1'b0;
                                if (tile_is_last) begin
                                    pp_rd_release <= 1'b1;
                                    e_state       <= E_DRAIN;
                                end
                            end else begin
                                sc_d <= sc_d + 1'b1;
                            end
                        end
                    end else begin
                        // ---- CONTEXT: one pass per N output dims ----
                        if (cx_go) begin
                            if (u_last) begin
                                cx_t <= 8'd0;
                                if (e_pass == PASSES - 1) begin
                                    pp_rd_release <= 1'b1;
                                    e_state       <= E_DRAIN;
                                end else begin
                                    e_pass  <= e_pass + 1'b1;
                                    e_state <= E_NEXT;
                                end
                            end else begin
                                cx_t <= cx_t + 1'b1;
                            end
                        end
                    end
                end

                E_NEXT: begin
                    if (!kv_rd_busy) begin
                        kv_rd_start <= 1'b1;
                        e_state     <= E_RUN;
                    end
                end

                E_DRAIN: begin
                    if (pending == 2'd0 && !cap_full_q && !hold && !meta_push)
                        e_state <= E_IDLE;
                end

                default: e_state <= E_IDLE;
            endcase
        end
    end

    assign busy = !eng_idle || (p_state != P_HDR);

endmodule
