#include <vector>
#include <type_traits>
#include <cmath>
#include <cstring>
#include <cstdint>

#include "ggml-bitnet.h"
#include "ggml-quants.h"

#define QK_I2_S 128
#define QK_I2 128

#if defined(__riscv) || defined(__riscv__)

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

// Horizontal sum fallback
static inline int hsum_i32_8(const int32_t *v) {
    int sum = 0;
    for (int i = 0; i < 8; ++i) sum += v[i];
    return sum;
}

// Quantization to 2-bit symmetric representation (I2_S)
size_t quantize_i2_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);
    int64_t n = nrow * n_per_row;

    float max = 0.0f;

#if defined(__riscv_vector)
    // RVV max absolute value
    size_t i = 0, vl;
    for (; i < n;) {
        vl = vsetvl_e32m1(n - i);
        vfloat32m1_t vsrc = vle32_v_f32m1(&src[i], vl);
        vfloat32m1_t vabs = vfabs_v_f32m1(vsrc, vl);
        float vmax_local = vfredmax_vs_f32m1_f32(vabs, vfmv_s_f_f32m1(vabs, 0.0f), vl);
        max = std::max(max, vmax_local);
        i += vl;
    }
#else
    for (int64_t i = 0; i < n; ++i) {
        max = std::max(max, std::fabs(src[i]));
    }
#endif

    float i2_scale = (max == 0.0f) ? 1.0f : max;

    // Allocate and fill temporary quantized buffer
    uint8_t *q2 = (uint8_t *)malloc(n * sizeof(uint8_t));
#if defined(__riscv_vector)
    i = 0;
    for (; i < n;) {
        vl = vsetvl_e32m1(n - i);
        vfloat32m1_t vsrc = vle32_v_f32m1(&src[i], vl);
        vfloat32m1_t vscaled = vfmul_vf_f32m1(vsrc, 1.0f / i2_scale, vl);

        // sign: + → 2, 0 → 1, - → 0
        vint32m1_t vsign = vsgt_vx_i32m1(vscaled, 0.0f, vl);
        vint32m1_t vzero = vmseq_vx_i32m1(vscaled, 0.0f, vl);
        vint32m1_t vq2 = vmadd_vx_i32m1(vsign, 2, vl);
        vq2 = vmerge_vvm_i32m1(vq2, vzero, vzero, vl); // if zero, set to 1

        vse32_v_i32m1((int32_t*)&q2[i], vq2, vl);  // Still 32-bit packed
        i += vl;
    }
    // Downcast q2[i] = int to uint8 for now (inefficient but simple)
    for (int64_t j = 0; j < n; ++j) q2[j] &= 0x03;
#else
    for (int64_t i = 0; i < n; ++i) {
        if (std::fabs(src[i]) < 1e-6f) {
            q2[i] = 1;
        } else {
            q2[i] = src[i] * i2_scale > 0 ? 2 : 0;
        }
    }
#endif

    memset(dst, 0, row_size * nrow);

    uint8_t *i2_weight = (uint8_t *)dst;
    for (int64_t i = 0; i < n / QK_I2; ++i) {
        for (int j = 0; j < QK_I2; ++j) {
            int group_idx = j / 32;
            int group_pos = j % 32;
            uint8_t val = q2[i * QK_I2 + j];
            int shift = (j % 4) * 2;
            i2_weight[i * 32 + group_idx * 8 + (j / 4)] |= (val << shift);
        }
    }

    float *scale_ptr = (float *)((char *)i2_weight + row_size * nrow - sizeof(float));
    scale_ptr[0] = i2_scale;

    free(q2);
    return row_size * nrow;
}

// Dot product of I2 (2-bit) * I8 (8-bit)
void ggml_vec_dot_i2_i8_s(
    int n,
    float * s,
    size_t bs,
    const void * vx,
    size_t bx,
    const void * vy,
    size_t by,
    int nrc) {

    const uint8_t *x = (const uint8_t *)vx;
    const int8_t *y = (const int8_t *)vy;

#if defined(__riscv_vector)
    int i = 0;
    int32_t total = 0;
    size_t vl;
    for (; i < n; ) {
        vl = vsetvl_e8m1(n - i);

        vint8m1_t vyvec = vle8_v_i8m1(&y[i], vl);

        // Unpack 4 2-bit values per byte from x[]
        int8_t unpacked[256];  // conservative for vl=256
        for (size_t j = 0; j < vl; ++j) {
            uint8_t packed = x[(i + j) / 4];
            int shift = (3 - ((i + j) % 4)) * 2;
            int val = ((packed >> shift) & 0x03) - 1;  // Map 0→-1, 1→0, 2→+1
            unpacked[j] = val;
        }

        vint8m1_t vxvec = vle8_v_i8m1((const int8_t*)unpacked, vl);
        vint16m2_t prod = vmul_vv_i16m2(vzext_vf8m1_i16m2(vxvec, vl), vzext_vf8m1_i16m2(vyvec, vl), vl);
        vint32m4_t sum = vwadd_vv_i32m4(vundefined_i32m4(), prod, vl);
        total += vfredsum_vs_i32m4_i32m1(sum, vmv_s_x_i32m1(vundefined_i32m1(), 0), vl)[0];

        i += vl;
    }

    *s = static_cast<float>(total);
#else
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        uint8_t xi = x[i / 4];
        int shift = (3 - (i % 4)) * 2;
        int xval = ((xi >> shift) & 0x03) - 1;
        sum += xval * y[i];
    }
    *s = static_cast<float>(sum);
#endif
}

#endif // __riscv