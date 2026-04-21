/*============================================================================
 * main.c -- Nios II Firmware for MNIST TinyML SoC
 *============================================================================
 * Firmware nay thuc hien:
 *   1. Doc anh test da nhung san trong test_image.h (784 bytes, da quantize)
 *   2. Ghi 784 pixel vao MLP Accelerator qua Avalon-MM bus
 *   3. Gui lenh START de trigger inference
 *   4. Doi phan cung tinh xong (~2ms)
 *   5. Doc ket qua tu thanh ghi RESULT
 *   6. In ket qua ra JTAG UART (hien thi tren Nios II Eclipse Console)
 *   7. HEX0 tu dong hien thi ket qua (direct hardware, khong can software)
 *
 * Build: Nios II Eclipse (SBT) voi HAL BSP
 * Target: DE10-Standard (Cyclone V)
 *============================================================================*/

#include "system.h"
#include "io.h"
#include "alt_types.h"
#include "altera_avalon_jtag_uart_regs.h"

/* Anh test nhung san (784 bytes, da quantize [0, 127]) */
#include "test_image.h"

/*--- MLP Accelerator Register Offsets (byte address) ---*/
#define MLP_REG_CTRL     0x0000    /* W: bit0=START, bit1=SOFT_RESET  */
#define MLP_REG_STATUS   0x0004    /* R: bit0=BUSY, bit1=DONE         */
#define MLP_REG_RESULT   0x0008    /* R: predicted digit 0-9           */
#define MLP_REG_SCORE    0x000C    /* R: max score (INT32)              */
#define MLP_REG_OUTPUT   0x0010    /* R: output[0]..output[9] (x4B)    */
#define MLP_REG_INPUT    0x1000    /* W: pixel[0]..pixel[783] (x4B)    */

#define MLP_CTRL_START   0x01
#define MLP_CTRL_RESET   0x02
#define MLP_STATUS_BUSY  0x01
#define MLP_STATUS_DONE  0x02

/*--- JTAG UART Helpers (khong can printf/getc, tuong thich Small C Lib) ---*/

static void jtag_uart_putchar(alt_u32 base, alt_u8 ch)
{
    alt_u32 ctrl;
    do {
        ctrl = IORD_ALTERA_AVALON_JTAG_UART_CONTROL(base);
    } while ((ctrl & ALTERA_AVALON_JTAG_UART_CONTROL_WSPACE_MSK) == 0);
    IOWR_ALTERA_AVALON_JTAG_UART_DATA(base, ch);
}

static void jtag_print(alt_u32 base, const char *str)
{
    while (*str) {
        jtag_uart_putchar(base, (alt_u8)*str++);
    }
}

static void jtag_print_dec(alt_u32 base, int val)
{
    char buf[12];
    int i = 0;
    int neg = 0;

    if (val < 0) { neg = 1; val = -val; }
    if (val == 0) {
        buf[i++] = '0';
    } else {
        while (val > 0) {
            buf[i++] = '0' + (val % 10);
            val /= 10;
        }
    }
    if (neg) buf[i++] = '-';
    while (i > 0) jtag_uart_putchar(base, (alt_u8)buf[--i]);
}

/*--- MLP Accelerator Driver ---*/

static inline void mlp_soft_reset(alt_u32 base)
{
    IOWR_32DIRECT(base, MLP_REG_CTRL, MLP_CTRL_RESET);
}

static inline void mlp_write_pixel(alt_u32 base, int index, alt_u8 value)
{
    IOWR_32DIRECT(base, MLP_REG_INPUT + (index << 2), (alt_u32)value);
}

static inline void mlp_start(alt_u32 base)
{
    IOWR_32DIRECT(base, MLP_REG_CTRL, MLP_CTRL_START);
}

static inline int mlp_is_done(alt_u32 base)
{
    return (IORD_32DIRECT(base, MLP_REG_STATUS) & MLP_STATUS_DONE) != 0;
}

static inline int mlp_get_result(alt_u32 base)
{
    return (int)(IORD_32DIRECT(base, MLP_REG_RESULT) & 0x0F);
}

static inline alt_32 mlp_get_score(alt_u32 base, int digit)
{
    return (alt_32)IORD_32DIRECT(base, MLP_REG_OUTPUT + (digit << 2));
}

/*============================================================================
 * Main
 *============================================================================
 * Luong xu ly:
 *
 *   test_image.h (784 bytes)
 *         |
 *         v
 *   [Ghi vao MLP Acc qua Avalon-MM]  --> input_buf trong phan cung
 *         |
 *         v
 *   [Gui START]                       --> FSM bat dau tinh
 *         |
 *         v
 *   [Doi DONE]                        --> ~102K cycles (~2ms @ 50MHz)
 *         |
 *         +---> [HEX0]  (truc tiep tu phan cung, khong can SW)
 *         |
 *         v
 *   [Doc RESULT]                      --> Doc tu thanh ghi
 *         |
 *         v
 *   [In ra JTAG UART]                 --> Hien thi tren Eclipse Console
 *
 *============================================================================*/

int main(void)
{
    /* Base addresses (tu system.h, duoc Platform Designer generate) */
#ifdef MLP_ACCELERATOR_0_BASE
    const alt_u32 mlp_base = MLP_ACCELERATOR_0_BASE;
#else
    const alt_u32 mlp_base = 0x00010000;  /* Fallback */
#endif

#ifdef JTAG_UART_0_BASE
    const alt_u32 uart_base = JTAG_UART_0_BASE;
#else
    const alt_u32 uart_base = 0x00041000;  /* Fallback */
#endif

    int digit, i;
    alt_32 max_score;

    /* --- Banner --- */
    jtag_print(uart_base, "\r\n");
    jtag_print(uart_base, "================================================\r\n");
    jtag_print(uart_base, "  MNIST Digit Recognition -- TinyML SoC\r\n");
    jtag_print(uart_base, "  MLP: 784 -> 128 (Leaky ReLU) -> 10\r\n");
    jtag_print(uart_base, "  Board: DE10-Standard (Cyclone V)\r\n");
    jtag_print(uart_base, "================================================\r\n\r\n");

    /* --- Step 1: Reset accelerator --- */
    jtag_print(uart_base, "[1] Resetting MLP Accelerator...\r\n");
    mlp_soft_reset(mlp_base);

    /* --- Step 2: Ghi 784 pixels tu test_image.h vao accelerator --- */
    jtag_print(uart_base, "[2] Writing test image (784 pixels)...\r\n");

    /*
     * LUU Y: Anh trong test_image.h DA DUOC quantize [0, 127]
     * boi notebook training (input.mif).
     * KHONG can normalize them (khong can >> 1).
     * Ghi truc tiep vao input buffer cua accelerator.
     */
    for (i = 0; i < 784; i++) {
        mlp_write_pixel(mlp_base, i, test_image[i]);
    }
    jtag_print(uart_base, "    Image loaded successfully.\r\n");

    /* --- Step 3: Trigger START --- */
    jtag_print(uart_base, "[3] Starting inference...\r\n");
    mlp_start(mlp_base);

    /* --- Step 4: Doi phan cung tinh xong --- */
    jtag_print(uart_base, "    Waiting for hardware...\r\n");
    while (!mlp_is_done(mlp_base)) {
        /* Busy-wait: ~102K cycles = ~2ms @ 50MHz */
    }
    jtag_print(uart_base, "    Inference complete!\r\n");

    /* --- Step 5: Doc ket qua --- */
    digit = mlp_get_result(mlp_base);
    max_score = mlp_get_score(mlp_base, digit);

    /* --- Step 6: In ket qua ra Eclipse Console (JTAG UART) --- */
    jtag_print(uart_base, "\r\n");
    jtag_print(uart_base, "================================================\r\n");
    jtag_print(uart_base, "  RESULT\r\n");
    jtag_print(uart_base, "================================================\r\n");
    jtag_print(uart_base, "\r\n");
    jtag_print(uart_base, "  Predicted digit: ");
    jtag_print_dec(uart_base, digit);
    jtag_print(uart_base, "\r\n");
    jtag_print(uart_base, "  Max score:       ");
    jtag_print_dec(uart_base, (int)max_score);
    jtag_print(uart_base, "\r\n\r\n");

    /* In scores cua tat ca 10 digits */
    jtag_print(uart_base, "  Output scores:\r\n");
    for (i = 0; i < 10; i++) {
        jtag_print(uart_base, "    Digit ");
        jtag_print_dec(uart_base, i);
        jtag_print(uart_base, ": ");
        jtag_print_dec(uart_base, (int)mlp_get_score(mlp_base, i));
        if (i == digit) {
            jtag_print(uart_base, "  <-- PREDICTED");
        }
        jtag_print(uart_base, "\r\n");
    }

    jtag_print(uart_base, "\r\n");
    jtag_print(uart_base, "================================================\r\n");
    jtag_print(uart_base, "  HEX0 is now displaying: ");
    jtag_print_dec(uart_base, digit);
    jtag_print(uart_base, "\r\n");
    jtag_print(uart_base, "  (HEX0 updated by hardware, no SW needed)\r\n");
    jtag_print(uart_base, "================================================\r\n");

    /* --- Step 7: Bat LED tuong ung voi digit --- */
#ifdef LED_BASE
    IOWR_32DIRECT(LED_BASE, 0, (1 << digit));
#endif

    /* Dung tai day — ket qua hien thi tren HEX0 + Console */
    while (1) {
        /* Halt — ket qua da hien thi */
    }

    return 0;
}
