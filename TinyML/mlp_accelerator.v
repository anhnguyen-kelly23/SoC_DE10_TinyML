//============================================================================
// mlp_accelerator.v — TinyML MLP Accelerator with Avalon-MM Interface
//============================================================================
// MNIST Handwritten Digit Recognition: MLP 784 → 128 (Leaky ReLU) → 10
//
// Features:
//   - Avalon-MM Slave interface (for Nios II / Platform Designer)
//   - INT8 weights, INT32 accumulator
//   - On-chip BRAM for weights (W1, W2) and biases (b1)
//   - Sequential MAC (1 DSP block), pipelined 1 MAC/cycle
//   - Hardware Leaky ReLU (α=0.125), requantization, argmax
//   - ~102K cycles inference (~2ms @ 50MHz)
//
// Register Map (word address / byte address):
//   0x000 / 0x0000 : CTRL     (W) bit0=START, bit1=SOFT_RESET
//   0x001 / 0x0004 : STATUS   (R) bit0=BUSY, bit1=DONE
//   0x002 / 0x0008 : RESULT   (R) predicted digit 0-9
//   0x003 / 0x000C : SCORE    (R) max score (INT32)
//   0x004 / 0x0010 : OUT[0]   (R) score for digit 0
//   ...
//   0x00D / 0x0034 : OUT[9]   (R) score for digit 9
//   0x400 / 0x1000 : INPUT[0] (W) pixel 0 (bits [7:0])
//   ...
//   0x70F / 0x1C3C : INPUT[783](W) pixel 783
//
// Target: Terasic DE10-Standard, Cyclone V (5CSXFC6D6F31C6)
//============================================================================

module mlp_accelerator #(
    parameter INPUT_SIZE       = 784,         // 28×28 pixels
    parameter HIDDEN_SIZE      = 128,         // Hidden layer neurons
    parameter OUTPUT_SIZE      = 10,          // Digits 0-9
    parameter W1_DEPTH         = 100352,      // INPUT_SIZE × HIDDEN_SIZE
    parameter W2_DEPTH         = 1280,        // HIDDEN_SIZE × OUTPUT_SIZE
    parameter L1_REQUANT_MULT  = 16'd12,      // Requantization multiplier (from Python notebook)
    parameter L1_REQUANT_SHIFT = 5'd15,       // Requantization shift (from Python notebook)
    parameter LEAKY_SHIFT      = 3,           // Leaky ReLU: alpha = 1/2^3 = 0.125
    parameter W1_INIT_FILE     = "w1_int8.hex",
    parameter W2_INIT_FILE     = "w2_int8.hex",
    parameter B1_INIT_FILE     = "b1_int32.hex"
)(
    // Avalon-MM Slave Interface
    input  wire        clk,
    input  wire        reset,               // Active-high synchronous reset
    input  wire [10:0] avs_address,          // Word address (11 bits = 2048 words)
    input  wire        avs_read,
    input  wire        avs_write,
    input  wire [31:0] avs_writedata,
    output reg  [31:0] avs_readdata,

    // Direct outputs
    output wire [3:0]  result_digit,         // Predicted digit (direct)
    output wire        result_valid,         // Inference complete flag
    output wire        irq                   // Interrupt on completion
);

    // ========================================================================
    // FSM States
    // ========================================================================
    localparam [3:0]
        S_IDLE     = 4'd0,
        S_L1_INIT  = 4'd1,   // Clear MAC, issue b1 address
        S_L1_BIAS  = 4'd2,   // Load bias into MAC, issue first input/weight addr
        S_L1_MAC   = 4'd3,   // MAC loop for layer 1 (pipelined)
        S_L1_POST  = 4'd4,   // Requantize + ReLU + store to hidden buffer
        S_L2_INIT  = 4'd5,   // Clear MAC, issue first hidden/w2 addr
        S_L2_MAC   = 4'd6,   // MAC loop for layer 2 (pipelined)
        S_L2_POST  = 4'd7,   // Store output
        S_ARGMAX   = 4'd8,   // Find maximum output
        S_DONE     = 4'd9;   // Inference complete

    // ========================================================================
    // Internal Registers
    // ========================================================================
    reg  [3:0]  state;
    reg  [7:0]  neuron_idx;        // Current neuron index
    reg  [9:0]  mac_counter;       // MAC iteration counter (max 784)
    reg  [3:0]  argmax_counter;    // Argmax iteration counter

    // Weight ROM address tracking (auto-incrementing)
    reg  [16:0] w1_addr_reg;       // Address into W1 ROM (0 to 100351)
    reg  [10:0] w2_addr_reg;       // Address into W2 ROM (0 to 1279)

    // Control/Status
    reg         start_pulse;
    reg         busy;
    reg         done_flag;

    // Result
    reg  [3:0]  result_reg;
    reg  signed [31:0] max_score_reg;
    reg  signed [31:0] best_score;
    reg  [3:0]  best_digit;

    // Output scores
    reg  signed [31:0] output_reg [0:OUTPUT_SIZE-1];

    // Assign direct outputs
    assign result_digit = result_reg;
    assign result_valid = done_flag;
    assign irq          = done_flag;

    // ========================================================================
    // Input Buffer (784 × 8-bit, BRAM)
    // ========================================================================
    (* ramstyle = "M10K" *)
    reg signed [7:0] input_buf [0:INPUT_SIZE-1];
    reg signed [7:0] input_buf_rdata;
    reg [9:0]  input_buf_raddr;        // Combinational read address

    // Input buffer write (from Avalon-MM)
    wire input_buf_we = avs_write && !busy &&
                        (avs_address >= 11'h400) &&
                        (avs_address < (11'h400 + INPUT_SIZE));
    wire [9:0] input_buf_waddr = avs_address[9:0];  // Lower 10 bits

    always @(posedge clk) begin
        if (input_buf_we)
            input_buf[input_buf_waddr] <= avs_writedata[7:0];
    end

    // Input buffer read (synchronous, 1-cycle latency)
    always @(posedge clk) begin
        input_buf_rdata <= input_buf[input_buf_raddr];
    end

    // ========================================================================
    // Hidden Buffer (128 × 8-bit)
    // ========================================================================
    reg signed [7:0] hidden_buf [0:HIDDEN_SIZE-1];
    reg signed [7:0] hidden_buf_rdata;
    reg [6:0]  hidden_buf_raddr;       // Combinational read address

    // Hidden buffer write control
    reg        hidden_buf_we;
    reg [6:0]  hidden_buf_waddr;
    reg signed [7:0] hidden_buf_wdata;

    always @(posedge clk) begin
        if (hidden_buf_we)
            hidden_buf[hidden_buf_waddr] <= hidden_buf_wdata;
    end

    always @(posedge clk) begin
        hidden_buf_rdata <= hidden_buf[hidden_buf_raddr];
    end

    // ========================================================================
    // Weight ROMs (instantiated)
    // ========================================================================
    reg  [16:0] w1_rom_addr;       // Combinational
    wire signed [7:0] w1_rom_data;

    weight_rom #(
        .DEPTH      (W1_DEPTH),
        .DATA_WIDTH (8),
        .ADDR_WIDTH (17),
        .INIT_FILE  (W1_INIT_FILE)
    ) u_w1_rom (
        .clk  (clk),
        .addr (w1_rom_addr),
        .data (w1_rom_data)
    );

    reg  [10:0] w2_rom_addr;       // Combinational
    wire signed [7:0] w2_rom_data;

    weight_rom #(
        .DEPTH      (W2_DEPTH),
        .DATA_WIDTH (8),
        .ADDR_WIDTH (11),
        .INIT_FILE  (W2_INIT_FILE)
    ) u_w2_rom (
        .clk  (clk),
        .addr (w2_rom_addr),
        .data (w2_rom_data)
    );

    reg  [6:0]  b1_rom_addr;       // Combinational
    wire signed [31:0] b1_rom_data;

    weight_rom #(
        .DEPTH      (HIDDEN_SIZE),
        .DATA_WIDTH (32),
        .ADDR_WIDTH (7),
        .INIT_FILE  (B1_INIT_FILE)
    ) u_b1_rom (
        .clk  (clk),
        .addr (b1_rom_addr),
        .data (b1_rom_data)
    );

    // ========================================================================
    // MAC Unit
    // ========================================================================
    reg         mac_clear;
    reg         mac_load_en;
    reg         mac_en;
    wire signed [31:0] mac_acc;

    // MUX: select MAC inputs based on current layer
    reg signed [7:0] mac_a;     // Activation input
    reg signed [7:0] mac_b;     // Weight input

    mac_unit u_mac (
        .clk      (clk),
        .rst_n    (~reset),
        .clear    (mac_clear),
        .load_en  (mac_load_en),
        .load_val (b1_rom_data),
        .mac_en   (mac_en),
        .a        (mac_a),
        .b        (mac_b),
        .acc      (mac_acc)
    );

    // ========================================================================
    // Requantization + Leaky ReLU (Combinational)
    // ========================================================================
    wire signed [15:0] requant_mult_s = L1_REQUANT_MULT;
    wire signed [47:0] requant_product = mac_acc * requant_mult_s;
    wire signed [47:0] requant_shifted = requant_product >>> L1_REQUANT_SHIFT;

    // Clamp to INT8 range [-127, 127] (matches Python: np.clip(..., -127, 127))
    wire signed [7:0] requant_clamped;
    assign requant_clamped = (requant_shifted > 48'sd127)  ? 8'sd127 :
                             (requant_shifted < -48'sd127) ? -8'sd127 :
                             requant_shifted[7:0];

    // Leaky ReLU: x >= 0 ? x : x >>> LEAKY_SHIFT  (alpha = 1/8 = 0.125)
    // Matches Python: np.where(z1_scaled >= 0, z1_scaled, z1_scaled >> 3)
    wire signed [7:0] leaky_relu_output;
    assign leaky_relu_output = requant_clamped[7] ? (requant_clamped >>> LEAKY_SHIFT) : requant_clamped;

    // ========================================================================
    // Combinational Address Generation
    // ========================================================================
    always @(*) begin
        // Defaults
        b1_rom_addr      = neuron_idx[6:0];
        input_buf_raddr  = 10'd0;
        w1_rom_addr      = w1_addr_reg;
        hidden_buf_raddr = 7'd0;
        w2_rom_addr      = w2_addr_reg;

        case (state)
            S_L1_INIT: begin
                b1_rom_addr = neuron_idx[6:0];
            end
            S_L1_BIAS: begin
                input_buf_raddr = 10'd0;
                w1_rom_addr     = w1_addr_reg;
            end
            S_L1_MAC: begin
                if (mac_counter < INPUT_SIZE)
                    input_buf_raddr = mac_counter[9:0];
                else
                    input_buf_raddr = 10'd0;
                w1_rom_addr = w1_addr_reg;
            end
            S_L2_INIT: begin
                hidden_buf_raddr = 7'd0;
                w2_rom_addr      = w2_addr_reg;
            end
            S_L2_MAC: begin
                if (mac_counter < HIDDEN_SIZE)
                    hidden_buf_raddr = mac_counter[6:0];
                else
                    hidden_buf_raddr = 7'd0;
                w2_rom_addr = w2_addr_reg;
            end
            default: ;
        endcase
    end

    // ========================================================================
    // Combinational MAC Control Signals
    // ========================================================================
    always @(*) begin
        mac_clear   = 1'b0;
        mac_load_en = 1'b0;
        mac_en      = 1'b0;
        mac_a       = 8'sd0;
        mac_b       = 8'sd0;

        case (state)
            S_L1_INIT: begin
                mac_clear = 1'b1;
            end
            S_L1_BIAS: begin
                mac_load_en = 1'b1;      // Load b1 data into accumulator
            end
            S_L1_MAC: begin
                if (mac_counter >= 10'd1) begin
                    mac_en = 1'b1;
                    mac_a  = input_buf_rdata;
                    mac_b  = w1_rom_data;
                end
            end
            S_L2_INIT: begin
                mac_clear = 1'b1;
            end
            S_L2_MAC: begin
                if (mac_counter >= 10'd1) begin
                    mac_en = 1'b1;
                    mac_a  = hidden_buf_rdata;
                    mac_b  = w2_rom_data;
                end
            end
            default: ;
        endcase
    end

    // ========================================================================
    // Hidden Buffer Write Control (Combinational)
    // ========================================================================
    always @(*) begin
        hidden_buf_we    = 1'b0;
        hidden_buf_waddr = neuron_idx[6:0];
        hidden_buf_wdata = leaky_relu_output;

        if (state == S_L1_POST) begin
            hidden_buf_we = 1'b1;
        end
    end

    // ========================================================================
    // Main FSM (Sequential)
    // ========================================================================
    integer k;

    always @(posedge clk) begin
        if (reset) begin
            state          <= S_IDLE;
            busy           <= 1'b0;
            done_flag      <= 1'b0;
            start_pulse    <= 1'b0;
            neuron_idx     <= 8'd0;
            mac_counter    <= 10'd0;
            argmax_counter <= 4'd0;
            w1_addr_reg    <= 17'd0;
            w2_addr_reg    <= 11'd0;
            result_reg     <= 4'd0;
            max_score_reg  <= 32'sd0;
            best_score     <= 32'sh80000000;  // Most negative
            best_digit     <= 4'd0;
            for (k = 0; k < OUTPUT_SIZE; k = k + 1)
                output_reg[k] <= 32'sd0;
        end else begin
            // One-shot start pulse
            start_pulse <= 1'b0;

            case (state)
                // --------------------------------------------------------
                S_IDLE: begin
                    if (start_pulse) begin
                        state       <= S_L1_INIT;
                        busy        <= 1'b1;
                        done_flag   <= 1'b0;
                        neuron_idx  <= 8'd0;
                        w1_addr_reg <= 17'd0;
                        w2_addr_reg <= 11'd0;
                    end
                end

                // --------------------------------------------------------
                // Layer 1: Hidden Layer (784 → 128, ReLU)
                // --------------------------------------------------------
                S_L1_INIT: begin
                    // b1_rom_addr = neuron_idx (combinational)
                    // BRAM reads b1[neuron_idx] → available next cycle
                    // MAC cleared (combinational mac_clear)
                    state <= S_L1_BIAS;
                end

                S_L1_BIAS: begin
                    // b1_rom_data now valid → loaded into MAC (combinational)
                    // input_buf_raddr=0, w1_rom_addr set (combinational)
                    // BRAM reads input[0] and w1[addr] → available next cycle
                    mac_counter <= 10'd1;
                    w1_addr_reg <= w1_addr_reg + 17'd1;
                    state       <= S_L1_MAC;
                end

                S_L1_MAC: begin
                    // mac_counter goes from 1 to INPUT_SIZE
                    // At counter=N (N>=1): MAC with data from addr set at counter=N-1
                    if (mac_counter < INPUT_SIZE) begin
                        // Issue next address (combinational addr = mac_counter)
                        w1_addr_reg <= w1_addr_reg + 17'd1;
                        mac_counter <= mac_counter + 10'd1;
                    end else begin
                        // mac_counter == INPUT_SIZE: last MAC done
                        state <= S_L1_POST;
                    end
                end

                S_L1_POST: begin
                    // hidden_buf write happens combinationally
                    // Advance to next neuron
                    if (neuron_idx == HIDDEN_SIZE - 1) begin
                        neuron_idx <= 8'd0;
                        state      <= S_L2_INIT;
                    end else begin
                        neuron_idx <= neuron_idx + 8'd1;
                        state      <= S_L1_INIT;
                    end
                end

                // --------------------------------------------------------
                // Layer 2: Output Layer (128 → 10, no bias)
                // --------------------------------------------------------
                S_L2_INIT: begin
                    // MAC cleared (combinational)
                    // hidden_buf_raddr=0, w2_rom_addr set (combinational)
                    // BRAM reads → available next cycle
                    mac_counter <= 10'd1;
                    w2_addr_reg <= w2_addr_reg + 11'd1;
                    state       <= S_L2_MAC;
                end

                S_L2_MAC: begin
                    if (mac_counter < HIDDEN_SIZE) begin
                        w2_addr_reg <= w2_addr_reg + 11'd1;
                        mac_counter <= mac_counter + 10'd1;
                    end else begin
                        state <= S_L2_POST;
                    end
                end

                S_L2_POST: begin
                    // Store raw INT32 output score
                    output_reg[neuron_idx[3:0]] <= mac_acc;

                    if (neuron_idx == OUTPUT_SIZE - 1) begin
                        neuron_idx     <= 8'd0;
                        argmax_counter <= 4'd0;
                        state          <= S_ARGMAX;
                    end else begin
                        neuron_idx <= neuron_idx + 8'd1;
                        state      <= S_L2_INIT;
                    end
                end

                // --------------------------------------------------------
                // Argmax: Find digit with highest score
                // --------------------------------------------------------
                S_ARGMAX: begin
                    if (argmax_counter == 4'd0) begin
                        best_digit     <= 4'd0;
                        best_score     <= output_reg[0];
                        argmax_counter <= 4'd1;
                    end else if (argmax_counter < OUTPUT_SIZE) begin
                        if (output_reg[argmax_counter] > best_score) begin
                            best_score <= output_reg[argmax_counter];
                            best_digit <= argmax_counter;
                        end
                        argmax_counter <= argmax_counter + 4'd1;
                    end else begin
                        result_reg    <= best_digit;
                        max_score_reg <= best_score;
                        state         <= S_DONE;
                    end
                end

                // --------------------------------------------------------
                S_DONE: begin
                    busy      <= 1'b0;
                    done_flag <= 1'b1;
                    // Stay until new start
                end

                default: state <= S_IDLE;
            endcase

            // ----- Avalon-MM Write Handling -----
            if (avs_write) begin
                if (avs_address == 11'h000) begin
                    if (avs_writedata[0]) start_pulse <= 1'b1;
                    if (avs_writedata[1]) begin
                        // Soft reset
                        state     <= S_IDLE;
                        busy      <= 1'b0;
                        done_flag <= 1'b0;
                    end
                end
            end
        end
    end

    // ========================================================================
    // Avalon-MM Read Handling (registered, readLatency = 1)
    // ========================================================================
    always @(posedge clk) begin
        if (reset) begin
            avs_readdata <= 32'h0;
        end else if (avs_read) begin
            case (avs_address)
                11'h000: avs_readdata <= {30'b0, 1'b0, start_pulse};
                11'h001: avs_readdata <= {30'b0, done_flag, busy};
                11'h002: avs_readdata <= {28'b0, result_reg};
                11'h003: avs_readdata <= max_score_reg;
                11'h004: avs_readdata <= output_reg[0];
                11'h005: avs_readdata <= output_reg[1];
                11'h006: avs_readdata <= output_reg[2];
                11'h007: avs_readdata <= output_reg[3];
                11'h008: avs_readdata <= output_reg[4];
                11'h009: avs_readdata <= output_reg[5];
                11'h00A: avs_readdata <= output_reg[6];
                11'h00B: avs_readdata <= output_reg[7];
                11'h00C: avs_readdata <= output_reg[8];
                11'h00D: avs_readdata <= output_reg[9];
                default: avs_readdata <= 32'h0;
            endcase
        end
    end

endmodule
