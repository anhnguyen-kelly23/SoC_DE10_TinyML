
module mnist_soc_top (
    // Clock & Reset
    input  wire        CLOCK_50,       // 50 MHz clock
    input  wire [3:0]  KEY,            // Push buttons (active-low)

    // Switches
    input  wire [9:0]  SW,

    // 7-Segment Displays (active-low)
    output wire [6:0]  HEX0,
    output wire [6:0]  HEX1,
    output wire [6:0]  HEX2,
    output wire [6:0]  HEX3,
    output wire [6:0]  HEX4,
    output wire [6:0]  HEX5
);

    wire sys_reset_n = KEY[0];
    wire [3:0] mlp_result_digit;  
    wire       mlp_result_valid;  


    system Nios_system (
        .clk_clk                              (CLOCK_50),
        .mlp_result_digit_result_digit        (mlp_result_digit),
        .mlp_result_valid_result_valid        (mlp_result_valid),
		  .reset_reset_n                        (sys_reset_n),
    );


    hex_decoder u_hex0 (
        .digit    (mlp_result_digit),
        .segments (HEX0)
    );

    assign HEX1 = 7'b1111111;
    assign HEX2 = 7'b1111111;
    assign HEX3 = 7'b1111111;
    assign HEX4 = 7'b1111111;
    assign HEX5 = 7'b1111111;

endmodule
