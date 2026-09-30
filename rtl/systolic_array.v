// -----------------------------------------------------------------------------
// systolic_array.v -- Parameterised N x N output-stationary INT8 systolic mesh
//
//   C[i][j] = sum_k A[i][k] * B[k][j]      (INT32 accumulation per PE)
//
//   Each cycle the caller presents one reduction step k:
//       a_in lane i = A[i][k]     (row operand, enters from the west)
//       b_in lane j = B[k][j]     (column operand, enters from the north)
//       in_valid, in_first (k==0), in_last (k==K-1)
//
//   Input skew: every lane first passes an ingress register; lane i of `a`
//   (and its control tags) is then delayed i more cycles and lane j of `b` j
//   more cycles, so A[i][k] and B[k][j] meet in PE(i,j) at cycle k+1+i+j.
//   Operands are zero-padded (forced to 0) whenever the beat
//   is not valid, so bubbles never inject stale data into the mesh.
//
//   out_valid pulses when PE(N-1,N-1) -- the last PE to finish -- latches its
//   result; at that instant all N*N results on `c_flat` belong to the same
//   tile. Latency from the in_last beat to out_valid is 2*(N-1) + 4 cycles.
//   Each PE holds its result until that PE receives the next tile's `last`.
//
//   c_flat packing: C[i][j] at bits [(i*N+j)*ACC_W +: ACC_W]
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module systolic_array #(
    parameter N      = 4,
    parameter DATA_W = 8,
    parameter ACC_W  = 32
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire [N*DATA_W-1:0]     a_in,
    input  wire [N*DATA_W-1:0]     b_in,
    input  wire                    in_valid,
    input  wire                    in_first,
    input  wire                    in_last,
    output wire [N*N*ACC_W-1:0]    c_flat,
    output wire                    out_valid
);

    // Mesh interconnect.  Horizontal buses have N+1 taps per row (col 0..N),
    // vertical buses have N+1 taps per column (row 0..N).
    wire [DATA_W-1:0] a_h [0:N*(N+1)-1];
    wire              v_h [0:N*(N+1)-1];
    wire              f_h [0:N*(N+1)-1];
    wire              l_h [0:N*(N+1)-1];
    wire [DATA_W-1:0] b_v [0:(N+1)*N-1];
    wire [N*N-1:0]    res_valid_vec;

    genvar i, j, s;

    // ---------------- west edge: ingress register + row skew + zero padding ----
    // Row i is delayed i+1 cycles: one ingress register (isolates the operand
    // SRAM / mux path from the PE multiplier) plus i skew registers.
    generate
        for (i = 0; i < N; i = i + 1) begin : g_row_skew
            wire [DATA_W-1:0] a_pad = in_valid ? a_in[i*DATA_W +: DATA_W] : {DATA_W{1'b0}};
            reg  [DATA_W-1:0] a_sr [0:i];
            reg  [i:0]        v_sr, f_sr, l_sr;
            always @(posedge clk) begin
                a_sr[0] <= a_pad;
            end
            for (s = 1; s <= i; s = s + 1) begin : g_a_stage
                always @(posedge clk) a_sr[s] <= a_sr[s-1];
            end
            always @(posedge clk) begin
                if (!rst_n) begin
                    v_sr <= {(i+1){1'b0}};
                    f_sr <= {(i+1){1'b0}};
                    l_sr <= {(i+1){1'b0}};
                end else begin
                    v_sr <= {v_sr, in_valid};              // width-truncating shift
                    f_sr <= {f_sr, in_valid & in_first};
                    l_sr <= {l_sr, in_valid & in_last};
                end
            end
            assign a_h[i*(N+1)] = a_sr[i];
            assign v_h[i*(N+1)] = v_sr[i];
            assign f_h[i*(N+1)] = f_sr[i];
            assign l_h[i*(N+1)] = l_sr[i];
        end
    endgenerate

    // ---------------- north edge: ingress register + column skew + zero padding
    generate
        for (j = 0; j < N; j = j + 1) begin : g_col_skew
            wire [DATA_W-1:0] b_pad = in_valid ? b_in[j*DATA_W +: DATA_W] : {DATA_W{1'b0}};
            reg  [DATA_W-1:0] b_sr [0:j];
            always @(posedge clk) begin
                b_sr[0] <= b_pad;
            end
            for (s = 1; s <= j; s = s + 1) begin : g_b_stage
                always @(posedge clk) b_sr[s] <= b_sr[s-1];
            end
            assign b_v[j] = b_sr[j];
        end
    endgenerate

    // ---------------- PE mesh ----------------
    generate
        for (i = 0; i < N; i = i + 1) begin : g_r
            for (j = 0; j < N; j = j + 1) begin : g_c
                wire [DATA_W-1:0] a_o, b_o;
                wire              v_o, f_o, l_o;
                wire [ACC_W-1:0]  r_o;
                wire              rv_o;

                pe #(
                    .DATA_W (DATA_W),
                    .ACC_W  (ACC_W)
                ) u_pe (
                    .clk       (clk),
                    .rst_n     (rst_n),
                    .a_in      (a_h[i*(N+1)+j]),
                    .b_in      (b_v[i*N+j]),
                    .v_in      (v_h[i*(N+1)+j]),
                    .f_in      (f_h[i*(N+1)+j]),
                    .l_in      (l_h[i*(N+1)+j]),
                    .a_out     (a_o),
                    .b_out     (b_o),
                    .v_out     (v_o),
                    .f_out     (f_o),
                    .l_out     (l_o),
                    .res       (r_o),
                    .res_valid (rv_o)
                );

                assign a_h[i*(N+1)+j+1] = a_o;
                assign v_h[i*(N+1)+j+1] = v_o;
                assign f_h[i*(N+1)+j+1] = f_o;
                assign l_h[i*(N+1)+j+1] = l_o;
                assign b_v[(i+1)*N+j]   = b_o;
                assign c_flat[(i*N+j)*ACC_W +: ACC_W] = r_o;
                assign res_valid_vec[i*N+j] = rv_o;
            end
        end
    endgenerate

    assign out_valid = res_valid_vec[N*N-1];

endmodule
