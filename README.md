# SoC DE10 TinyML — MNIST Handwritten Digit Recognition on FPGA

Hệ thống nhận dạng chữ số viết tay (MNIST) sử dụng mạng nơ-ron MLP lượng tử hóa INT8, được triển khai dưới dạng Custom IP tăng tốc phần cứng trên FPGA Intel Cyclone V (DE10-Standard). Toàn bộ pipeline từ huấn luyện mô hình, tổng hợp phần cứng đến firmware nhúng đều được tích hợp trong một repository.

---

## Kiến trúc tổng quan

```
┌──────────────────────────────────────────────────────────────┐
│                  PC — Huấn luyện (offline)                   │
│   mnist_mlp_training_and_export.ipynb                        │
│   ┌────────────────────────────────────────────────────┐     │
│   │  MNIST CSV  →  MLP Training (NumPy)  →  INT8 Quant │     │
│   │  42,000 mẫu    784→128→10 (Leaky ReLU)   FP32→INT8 │     │
│   └────────────────────────────────────────────────────┘     │
│           │                                                    │
│    w1_int8.hex   w2_int8.hex   b1_int32.hex   test_image.h   │
└───────────┼───────────────────────────────────────────────────┘
            │ (copy sang TinyML/)
┌───────────▼───────────────────────────────────────────────────┐
│               FPGA DE10-Standard (Cyclone V)                  │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Platform Designer (Qsys)                   │ │
│  │                                                         │ │
│  │  Nios II ──── Avalon-MM Bus ──── MLP Accelerator IP     │ │
│  │    CPU           │                   │                  │ │
│  │    │         JTAG UART          weight_rom (M10K)       │ │
│  │    │         On-chip RAM        mac_unit (1 DSP)        │ │
│  │    │         Timer                                      │ │
│  │    │                                                    │ │
│  │  Firmware                                               │ │
│  │  hello_world.c                                          │ │
│  │  test_image.h                                           │ │
│  └─────────────────────────────────────────────────────────┘ │
│                           │                                   │
│                    HEX0 Display                               │
│                    LEDR[9:0] LEDs                             │
│                    JTAG UART Console                          │
└───────────────────────────────────────────────────────────────┘
```

---

## Cấu trúc thư mục

```
SoC_DE10_TinyML/
│
├── MNIST/                              # Huấn luyện mô hình (chạy trên PC)
│   ├── mnist_mlp_training_and_export.ipynb  # Notebook chính: train + export
│   ├── test_image.h                    # 10 ảnh test (INT8) cho firmware
│   ├── w1_int8.hex                     # Trọng số Layer 1 (784×128, INT8)
│   ├── w2_int8.hex                     # Trọng số Layer 2 (128×10, INT8)
│   ├── b1_int32.hex                    # Bias Layer 1 (128, INT32)
│   ├── my_mnist_result.csv             # Kết quả suy luận FP32
│   ├── my_mnist_result_int8.csv        # Kết quả suy luận INT8
│   └── input/
│       ├── train.csv                   # 42,000 mẫu huấn luyện
│       └── test.csv                    # 28,000 mẫu kiểm tra
│
└── TinyML/                             # Phần cứng FPGA + Firmware Nios II
    ├── mlp_accelerator.v               # Custom IP chính (Avalon-MM slave)
    ├── mac_unit.v                      # Đơn vị MAC INT8 (1 DSP block)
    ├── weight_rom.v                    # ROM lưu trọng số (M10K BRAM)
    ├── hex_decoder.v                   # Decoder 7-segment cho HEX0
    ├── mnist_soc_top.v                 # Top-level kết nối board
    ├── tb_mlp_accelerator.v            # Testbench RTL
    ├── MLP_Accelerator_hw.tcl          # Định nghĩa Platform Designer IP
    ├── moduleTop.qsf                   # Cấu hình Quartus + pin assignments
    ├── moduleTop.qpf                   # File project Quartus
    ├── w1_int8.hex                     # Bản sao dùng trong synthesis
    ├── w2_int8.hex
    ├── b1_int32.hex
    ├── system/                         # Output Platform Designer (Qsys)
    │   ├── synthesis/system.v          # Top-level hệ thống đã tổng hợp
    │   └── synthesis/submodules/       # Các module con được mở rộng
    ├── output_files/                   # Output biên dịch Quartus (.sof, .rpt)
    └── Software/
        ├── MATiny/
        │   ├── hello_world.c           # Firmware chính Nios II
        │   └── test_image.h            # 10 ảnh test nhúng vào firmware
        └── MATiny_bsp/                 # Board Support Package (Altera HAL)
```

---

## Phần 1: MNIST — Huấn luyện mô hình TinyML

### Mô hình

| Tham số | Giá trị |
|---------|---------|
| Kiến trúc | MLP 784 → 128 → 10 |
| Hàm kích hoạt | Leaky ReLU (α = 0.125 = 1/8) |
| Optimizer | RMSprop + L2 Regularization |
| Dữ liệu huấn luyện | 42,000 ảnh MNIST (train.csv) |
| Độ chính xác FP32 | ~98.3% (training), ~100% (test set) |
| Độ chính xác INT8 | ~99.77% (sụt giảm 0.23%) |

### Quy trình lượng tử hóa (INT8 Symmetric Quantization)

- **Layer 1 weights**: FP32 → INT8, lưu theo thứ tự neuron-major
- **Layer 1 biases**: INT32 (= bias × input_scale × weight_scale)
- **Tham số re-quantization** (hardcode vào phần cứng):
  - `L1_REQUANT_MULT = 12`
  - `L1_REQUANT_SHIFT = 15`
  - `LEAKY_SHIFT = 3` (tương đương α = 1/8)

### Output files

| File | Mô tả | Kích thước |
|------|-------|-----------|
| `w1_int8.hex` | 100,352 giá trị INT8 (784×128) | 393 KB |
| `w2_int8.hex` | 1,280 giá trị INT8 (128×10) | 5.1 KB |
| `b1_int32.hex` | 128 giá trị INT32 | 1.3 KB |
| `test_image.h` | 10 ảnh test dạng `int8_t[10][784]` | 31 KB |

### Chạy notebook

```bash
cd MNIST/
jupyter notebook mnist_mlp_training_and_export.ipynb
```

Yêu cầu: Python 3.x, NumPy, Pandas, Jupyter.

---

## Phần 2: Hardware — Thiết kế FPGA

### Custom IP: MLP Accelerator

File chính: [TinyML/mlp_accelerator.v](TinyML/mlp_accelerator.v)

#### Giao diện Avalon-MM

| Địa chỉ | Register | Mô tả |
|---------|----------|-------|
| `0x000` | CTRL | Bit[0]: START, Bit[1]: SOFT_RESET |
| `0x004` | STATUS | Bit[0]: BUSY, Bit[1]: DONE |
| `0x008` | RESULT | Chữ số dự đoán (0–9) |
| `0x00C` | SCORE | Điểm số cao nhất (INT32) |
| `0x010`–`0x034` | OUT[0]–OUT[9] | 10 điểm đầu ra (INT32) |
| `0x1000`–`0x1C3C` | INPUT[0]–INPUT[783] | Buffer ảnh đầu vào (INT8) |

#### FSM Inference

```
IDLE → L1_INIT → L1_BIAS → L1_MAC (×784 pixels)
     → L1_POST (Leaky ReLU + Re-quantize)
     → L2_INIT → L2_MAC (×128 neurons)
     → L2_POST → ARGMAX → DONE
```

- **Latency**: ~102,048 cycles ≈ **2.04 ms @ 50 MHz**
- **DSP blocks**: 1 (MAC tuần tự, 1 phép tính/cycle)
- **BRAM**: ~110 M10K blocks (~100 KB tổng)

#### Các module phần cứng

| Module | File | Chức năng |
|--------|------|-----------|
| `mlp_accelerator` | [mlp_accelerator.v](TinyML/mlp_accelerator.v) | Custom IP chính, điều phối FSM |
| `mac_unit` | [mac_unit.v](TinyML/mac_unit.v) | INT8×INT8→INT32 accumulator |
| `weight_rom` | [weight_rom.v](TinyML/weight_rom.v) | ROM M10K, khởi tạo từ file HEX |
| `hex_decoder` | [hex_decoder.v](TinyML/hex_decoder.v) | Giải mã 7-segment (active-LOW) |
| `mnist_soc_top` | [mnist_soc_top.v](TinyML/mnist_soc_top.v) | Top-level kết nối với board |

### Platform Designer (Qsys)

Hệ thống Qsys gồm các thành phần:

```
Nios II Gen2 CPU
    └── Avalon-MM Master Bus
            ├── On-Chip Memory (RAM)
            ├── JTAG UART
            ├── Timer
            └── MLP Accelerator IP (Custom)
                    ├── mac_unit
                    └── weight_rom × 3 (W1, W2, B1)
```

File định nghĩa IP: [TinyML/MLP_Accelerator_hw.tcl](TinyML/MLP_Accelerator_hw.tcl)

### Tổng hợp với Quartus

1. Mở `TinyML/moduleTop.qpf` bằng Quartus Prime
2. Đảm bảo các file HEX (`w1_int8.hex`, `w2_int8.hex`, `b1_int32.hex`) nằm cùng thư mục `TinyML/`
3. **Processing → Start Compilation**
4. Nạp bitstream: **Tools → Programmer**, chọn file `output_files/*.sof`

---

## Phần 3: Firmware — Nios II trên Eclipse

### Luồng hoạt động

File chính: [TinyML/Software/MATiny/hello_world.c](TinyML/Software/MATiny/hello_world.c)

```
Khởi tạo hệ thống
    └── Vòng lặp 10 ảnh (từ test_image.h)
            ├── 1. Soft-reset MLP Accelerator
            ├── 2. Ghi 784 pixel qua Avalon-MM (địa chỉ 0x1000–0x1C3C)
            ├── 3. Ghi START (CTRL = 0x01)
            ├── 4. Polling STATUS cho đến khi DONE = 1
            ├── 5. Đọc RESULT (chữ số dự đoán)
            ├── 6. Đọc SCORE và OUT[0..9]
            ├── 7. In kết quả qua JTAG UART Console
            ├── 8. Hiển thị chữ số trên HEX0
            ├── 9. Bật LED[digit]
            └── 10. Chờ 5 giây → ảnh tiếp theo
```

### Output UART Console (mẫu)

```
=== Image 0 ===
Predicted digit: 7
Max score: 1234567
Scores: [  -234,  -456,  -678,  123,  -345,  -567,   -89,  1234, -901,  -123 ]
                                                               ^ predicted
```

### Nạp firmware lên board

1. Mở **Eclipse for Nios II** (đi kèm Quartus)
2. Import project `MATiny` và `MATiny_bsp`
3. Build BSP trước: chuột phải `MATiny_bsp` → **Nios II → Generate BSP**
4. Build project: chuột phải `MATiny` → **Build Project**
5. Nạp: **Run → Run Configurations → Nios II Hardware** → chọn board → Run
6. Xem kết quả: **Window → Show View → Nios II Console**

---

## Kết quả kiểm tra

Đã kiểm tra thành công 10 ảnh xuất ra từ `test_image.h` trên board DE10-Standard:

- Firmware chạy đúng luồng: ghi ảnh → kích hoạt accelerator → nhận kết quả
- Chữ số dự đoán hiển thị trên **HEX0**
- LED tương ứng với chữ số dự đoán sáng lên
- Kết quả in đầy đủ 10 output score qua **JTAG UART Console**

---

## Thông số kỹ thuật tóm tắt

| Thông số | Giá trị |
|---------|---------|
| FPGA | Intel Cyclone V SoC (5CSXFC6D6F31C6) |
| Board | DE10-Standard |
| Clock | 50 MHz |
| Kiến trúc mô hình | MLP 784→128→10 |
| Dạng dữ liệu | INT8 weights, INT32 accumulators |
| Thời gian suy luận | ~2.04 ms (102,048 cycles) |
| DSP blocks | 1 |
| BRAM | ~110 M10K blocks |
| Độ chính xác INT8 | ~99.77% |
| CPU nhúng | Nios II Gen2 |
| Giao tiếp IP | Avalon-MM Slave |
| Hiển thị kết quả | 7-segment HEX0 + LEDR + JTAG UART |

---

## Yêu cầu phần mềm

| Công cụ | Phiên bản khuyến nghị |
|---------|----------------------|
| Intel Quartus Prime | 18.1 trở lên |
| Nios II EDS / Eclipse | Đi kèm Quartus |
| ModelSim | Đi kèm Quartus (simulation) |
| Python | 3.8+ |
| Jupyter Notebook | Mới nhất |
| NumPy, Pandas | Mới nhất |

---

## Tác giả

- **Board**: DE10-Standard (Intel/Terasic)
- **Project**: Thực hành Hệ thống Nhúng — SoC TinyML
