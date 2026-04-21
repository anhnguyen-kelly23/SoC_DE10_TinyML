//============================================================================
// mac_unit.v — Multiply-Accumulate Unit for MLP Accelerator
//============================================================================
// INT8 × INT8 multiply, INT32 accumulate.
// Uses Cyclone V DSP 18×18 block for multiplication.
//
// Priority: clear > load > enable
//============================================================================

module mac_unit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clear,              // Clear accumulator to 0
    input  wire        load_en,            // Load external value into acc
    input  wire signed [31:0] load_val,    // Value to load
    input  wire        mac_en,             // Enable multiply-accumulate
    input  wire signed [7:0]  a,           // Input activation (INT8)
    input  wire signed [7:0]  b,           // Weight (INT8)
    output reg  signed [31:0] acc          // Accumulator output (INT32)
);

    // INT8 × INT8 = INT16 (signed multiplication)
    wire signed [15:0] product;
    assign product = a * b;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc <= 32'sd0;
        end else if (clear) begin
            acc <= 32'sd0;
        end else if (load_en) begin
            acc <= load_val;
        end else if (mac_en) begin
            // Sign-extend INT16 product to INT32 and accumulate
            acc <= acc + {{16{product[15]}}, product};
        end
    end

endmodule
