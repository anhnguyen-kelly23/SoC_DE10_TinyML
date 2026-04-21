transcript on
if {[file exists rtl_work]} {
	vdel -lib rtl_work -all
}
vlib rtl_work
vmap work rtl_work

vlog -vlog01compat -work work +incdir+D:/Quartus/TinyML {D:/Quartus/TinyML/weight_rom.v}
vlog -vlog01compat -work work +incdir+D:/Quartus/TinyML {D:/Quartus/TinyML/mlp_accelerator.v}
vlog -vlog01compat -work work +incdir+D:/Quartus/TinyML {D:/Quartus/TinyML/mac_unit.v}

vlog -vlog01compat -work work +incdir+D:/Quartus/TinyML {D:/Quartus/TinyML/tb_mlp_accelerator.v}

vsim -t 1ps -L altera_ver -L lpm_ver -L sgate_ver -L altera_mf_ver -L altera_lnsim_ver -L cyclonev_ver -L cyclonev_hssi_ver -L cyclonev_pcie_hip_ver -L rtl_work -L work -voptargs="+acc"  tb_mlp_accelerator

add wave *
view structure
view signals
run -all
