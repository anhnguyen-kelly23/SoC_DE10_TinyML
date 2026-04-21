//============================================================================
// weight_rom.v — Parameterized ROM for Neural Network Weights
//============================================================================
// Single-port synchronous ROM, inferred as Cyclone V M10K Block RAM.
// Initialized from Intel HEX file using $readmemh().
//
// Read latency: 1 clock cycle (registered output)
//============================================================================

module weight_rom #(
    parameter DEPTH      = 1024,
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 10,
    parameter INIT_FILE  = "weights.hex"
)(
    input  wire                         clk,
    input  wire [ADDR_WIDTH-1:0]        addr,
    output reg  signed [DATA_WIDTH-1:0] data
);

    // Infer M10K Block RAM
    (* ramstyle = "M10K" *)
    reg signed [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    // Initialize from hex file
    initial begin
        $readmemh(INIT_FILE, mem);
    end

    // Synchronous read (1 cycle latency)
    always @(posedge clk) begin
        data <= mem[addr];
    end

endmodule
