	component system is
		port (
			clk_clk                       : in  std_logic                    := 'X'; -- clk
			mlp_result_digit_result_digit : out std_logic_vector(3 downto 0);        -- result_digit
			mlp_result_valid_result_valid : out std_logic;                           -- result_valid
			reset_reset_n                 : in  std_logic                    := 'X'  -- reset_n
		);
	end component system;

	u0 : component system
		port map (
			clk_clk                       => CONNECTED_TO_clk_clk,                       --              clk.clk
			mlp_result_digit_result_digit => CONNECTED_TO_mlp_result_digit_result_digit, -- mlp_result_digit.result_digit
			mlp_result_valid_result_valid => CONNECTED_TO_mlp_result_valid_result_valid, -- mlp_result_valid.result_valid
			reset_reset_n                 => CONNECTED_TO_reset_reset_n                  --            reset.reset_n
		);

