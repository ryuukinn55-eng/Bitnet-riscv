#include <vector>
#include <type_traits>
#include <cmath>
#include <cstring>
#include <cstdint>

#include "ggml-bitnet.h"
#include "ggml-quants.h"

#define QK_I2_S 128
#define QK_I2 128
#define QK_I2_S_PACKED_BYTES 32
#define QK_I2_S_GROUP_SIZE (QK_I2_S / 4)

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#include <immintrin.h>
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__riscv_v_intrinsic) || defined(__riscv_vector)
#include <riscv_vector.h>
#endif

// Scalar fallback for a (possibly partial) 128-element I2_S block.
static inline int ggml_i2_s_block_dot_scalar(const uint8_t * x_block, const int8_t * y_block, int n) {
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        const uint8_t packed = x_block[i % QK_I2_S_PACKED_BYTES];
        const int shift = 6 - 2 * (i / QK_I2_S_PACKED_BYTES);
        const int xval = (packed >> shift) & 0x03;
        sum += xval * y_block[i];
    }
    return sum;
}

// Quantization to 2-bit symmetric representation (I2_S)
size_t quantize_i2_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);
    int64_t n = nrow * n_per_row;

    float max = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        max = std::max(max, std::fabs(src[i]));
    }

    float i2_scale = (max == 0.0f) ? 1.0f : max;

    uint8_t *q2 = (uint8_t *)malloc(n * sizeof(uint8_t));
    for (int64_t i = 0; i < n; ++i) {
        if (std::fabs(src[i]) < 1e-6f) {
            q2[i] = 1;
        } else {
            q2[i] = src[i] * i2_scale > 0 ? 2 : 0;
        }
    }

    memset(dst, 0, row_size * nrow);

    uint8_t *i2_weight = (uint8_t *)dst;
    const int64_t packed_row_size = n_per_row / 4;
    for (int64_t row = 0; row < nrow; ++row) {
        const int64_t row_offset = row * n_per_row;
        const int64_t packed_row_offset = row * packed_row_size;
        for (int64_t block = 0; block < n_per_row / QK_I2; ++block) {
            for (int j = 0; j < QK_I2; ++j) {
                const int group_idx = j / 32;
                const int group_pos = j % 32;
                const uint8_t val = q2[row_offset + block * QK_I2 + j];
                const int shift = 6 - 2 * group_idx;
                i2_weight[packed_row_offset + block * QK_I2_S_PACKED_BYTES + group_pos] |= (val << shift);
            }
        }
    }

    float *scale_ptr = (float *)((char *)i2_weight + packed_row_size * nrow);
    scale_ptr[0] = i2_scale;

    free(q2);
    return packed_row_size * nrow + 32;
}

#if defined(__riscv_v_intrinsic) || defined(__riscv_vector)
// RVV path. Two tuned variants, selected at runtime by VLEN:
//
//   * e8m1 path — VLEN>=256 (SpaceMit K1 X60, Sophgo SG2380, ...):
//     a 32-byte block fits exactly in one LMUL=1 group, so the vector unit
//     is fully utilized while using only half the registers of LMUL=2.
//     This is the fast path on BPI-F3.
//
//   * e8m2 path — VLEN=128 (T-Head C908, ...): a 32-byte block spans two
//     physical vector registers; LMUL=2 gives a single logical group.
//
// Both variants use 4 independent i16 accumulators (one per 2-bit quadrant)
// to break the vwmacc RAW chain — X60's vwmacc has ~4-6 cycle latency and
// pipelined 1/cycle throughput, so 4-way ILP is required to saturate.
//
// Per-lane i16 bound: |q * y| <= 2*127 = 254, so each accumulator stays
// below 254*CHUNK. CHUNK=64 gives 16256 < 32767 with comfortable margin.
#define BITNET_RVV_CHUNK 64

static inline int ggml_vec_dot_i2_i8_s_rvv_m1(int nb, const uint8_t * x, const int8_t * y) {
    // Caller guarantees VLMAX(e8m1) >= 32.
    const size_t vl = __riscv_vsetvl_e8m1(QK_I2_S_PACKED_BYTES);

    vint32m4_t accu32 = __riscv_vmv_v_x_i32m4(0, vl);

    int block = 0;
    while (block < nb) {
        int chunk = nb - block;
        if (chunk > BITNET_RVV_CHUNK) chunk = BITNET_RVV_CHUNK;

        vint16m2_t acc_a = __riscv_vmv_v_x_i16m2(0, vl);
        vint16m2_t acc_b = __riscv_vmv_v_x_i16m2(0, vl);
        vint16m2_t acc_c = __riscv_vmv_v_x_i16m2(0, vl);
        vint16m2_t acc_d = __riscv_vmv_v_x_i16m2(0, vl);

        for (int j = 0; j < chunk; ++j) {
            const uint8_t * xb = x + (block + j) * QK_I2_S_PACKED_BYTES;
            const int8_t  * yb = y + (block + j) * QK_I2_S;

            // K1 has a hardware stride prefetcher; a single hint 2 blocks
            // ahead is enough to keep L1 warm on the 128B/block y stream.
            if (j + 2 < chunk) {
                __builtin_prefetch(yb + 2 * QK_I2_S,      0, 0);
                __builtin_prefetch(yb + 2 * QK_I2_S + 64, 0, 0);
            }

            const vuint8m1_t packed = __riscv_vle8_v_u8m1(xb, vl);

            // q0 = packed >> 6  (top 2 bits, no mask needed)
            const vint8m1_t q0 = __riscv_vreinterpret_v_u8m1_i8m1(
                __riscv_vsrl_vx_u8m1(packed, 6, vl));
            const vint8m1_t q1 = __riscv_vreinterpret_v_u8m1_i8m1(
                __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(packed, 4, vl), 0x03, vl));
            const vint8m1_t q2 = __riscv_vreinterpret_v_u8m1_i8m1(
                __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(packed, 2, vl), 0x03, vl));
            const vint8m1_t q3 = __riscv_vreinterpret_v_u8m1_i8m1(
                __riscv_vand_vx_u8m1(packed, 0x03, vl));

            const vint8m1_t y0 = __riscv_vle8_v_i8m1(yb + 0 * QK_I2_S_GROUP_SIZE, vl);
            const vint8m1_t y1 = __riscv_vle8_v_i8m1(yb + 1 * QK_I2_S_GROUP_SIZE, vl);
            const vint8m1_t y2 = __riscv_vle8_v_i8m1(yb + 2 * QK_I2_S_GROUP_SIZE, vl);
            const vint8m1_t y3 = __riscv_vle8_v_i8m1(yb + 3 * QK_I2_S_GROUP_SIZE, vl);

            // Four parallel MAC chains — breaks X60's vwmacc RAW stall.
            acc_a = __riscv_vwmacc_vv_i16m2(acc_a, q0, y0, vl);
            acc_b = __riscv_vwmacc_vv_i16m2(acc_b, q1, y1, vl);
            acc_c = __riscv_vwmacc_vv_i16m2(acc_c, q2, y2, vl);
            acc_d = __riscv_vwmacc_vv_i16m2(acc_d, q3, y3, vl);
        }

        accu32 = __riscv_vwadd_wv_i32m4(accu32, acc_a, vl);
        accu32 = __riscv_vwadd_wv_i32m4(accu32, acc_b, vl);
        accu32 = __riscv_vwadd_wv_i32m4(accu32, acc_c, vl);
        accu32 = __riscv_vwadd_wv_i32m4(accu32, acc_d, vl);

        block += chunk;
    }

    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    const vint32m1_t sum  = __riscv_vredsum_vs_i32m4_i32m1(accu32, zero, vl);
    return __riscv_vmv_x_s_i32m1_i32(sum);
}

static inline int ggml_vec_dot_i2_i8_s_rvv_m2(int nb, const uint8_t * x, const int8_t * y) {
    // Caller guarantees VLMAX(e8m2) >= 32 (i.e. VLEN >= 128).
    const size_t vl = __riscv_vsetvl_e8m2(QK_I2_S_PACKED_BYTES);

    vint32m8_t accu32 = __riscv_vmv_v_x_i32m8(0, vl);

    int block = 0;
    while (block < nb) {
        int chunk = nb - block;
        if (chunk > BITNET_RVV_CHUNK) chunk = BITNET_RVV_CHUNK;

        vint16m4_t acc_a = __riscv_vmv_v_x_i16m4(0, vl);
        vint16m4_t acc_b = __riscv_vmv_v_x_i16m4(0, vl);
        vint16m4_t acc_c = __riscv_vmv_v_x_i16m4(0, vl);
        vint16m4_t acc_d = __riscv_vmv_v_x_i16m4(0, vl);

        for (int j = 0; j < chunk; ++j) {
            const uint8_t * xb = x + (block + j) * QK_I2_S_PACKED_BYTES;
            const int8_t  * yb = y + (block + j) * QK_I2_S;

            if (j + 2 < chunk) {
                __builtin_prefetch(yb + 2 * QK_I2_S,      0, 0);
                __builtin_prefetch(yb + 2 * QK_I2_S + 64, 0, 0);
            }

            const vuint8m2_t packed = __riscv_vle8_v_u8m2(xb, vl);

            const vint8m2_t q0 = __riscv_vreinterpret_v_u8m2_i8m2(
                __riscv_vsrl_vx_u8m2(packed, 6, vl));
            const vint8m2_t q1 = __riscv_vreinterpret_v_u8m2_i8m2(
                __riscv_vand_vx_u8m2(__riscv_vsrl_vx_u8m2(packed, 4, vl), 0x03, vl));
            const vint8m2_t q2 = __riscv_vreinterpret_v_u8m2_i8m2(
                __riscv_vand_vx_u8m2(__riscv_vsrl_vx_u8m2(packed, 2, vl), 0x03, vl));
            const vint8m2_t q3 = __riscv_vreinterpret_v_u8m2_i8m2(
                __riscv_vand_vx_u8m2(packed, 0x03, vl));

            const vint8m2_t y0 = __riscv_vle8_v_i8m2(yb + 0 * QK_I2_S_GROUP_SIZE, vl);
            const vint8m2_t y1 = __riscv_vle8_v_i8m2(yb + 1 * QK_I2_S_GROUP_SIZE, vl);
            const vint8m2_t y2 = __riscv_vle8_v_i8m2(yb + 2 * QK_I2_S_GROUP_SIZE, vl);
            const vint8m2_t y3 = __riscv_vle8_v_i8m2(yb + 3 * QK_I2_S_GROUP_SIZE, vl);

            acc_a = __riscv_vwmacc_vv_i16m4(acc_a, q0, y0, vl);
            acc_b = __riscv_vwmacc_vv_i16m4(acc_b, q1, y1, vl);
            acc_c = __riscv_vwmacc_vv_i16m4(acc_c, q2, y2, vl);
            acc_d = __riscv_vwmacc_vv_i16m4(acc_d, q3, y3, vl);
        }

        accu32 = __riscv_vwadd_wv_i32m8(accu32, acc_a, vl);
        accu32 = __riscv_vwadd_wv_i32m8(accu32, acc_b, vl);
        accu32 = __riscv_vwadd_wv_i32m8(accu32, acc_c, vl);
        accu32 = __riscv_vwadd_wv_i32m8(accu32, acc_d, vl);

        block += chunk;
    }

    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    const vint32m1_t sum  = __riscv_vredsum_vs_i32m8_i32m1(accu32, zero, vl);
    return __riscv_vmv_x_s_i32m1_i32(sum);
}

static inline int ggml_vec_dot_i2_i8_s_rvv(int nb, const uint8_t * x, const int8_t * y) {
    // Prefer e8m1: saturates VLEN>=256 and uses fewer registers.
    if (__riscv_vsetvl_e8m1(QK_I2_S_PACKED_BYTES) == (size_t)QK_I2_S_PACKED_BYTES) {
        return ggml_vec_dot_i2_i8_s_rvv_m1(nb, x, y);
    }
    // VLEN=128 path.
    if (__riscv_vsetvl_e8m2(QK_I2_S_PACKED_BYTES) == (size_t)QK_I2_S_PACKED_BYTES) {
        return ggml_vec_dot_i2_i8_s_rvv_m2(nb, x, y);
    }
    // VLEN too small for a 32-byte block in one group; scalar fallback.
    int total = 0;
    for (int b = 0; b < nb; ++b) {
        total += ggml_i2_s_block_dot_scalar(
            x + b * QK_I2_S_PACKED_BYTES, y + b * QK_I2_S, QK_I2_S);
    }
    return total;
}
#endif

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
    (void) bs;
    (void) bx;
    (void) by;
    (void) nrc;

    const uint8_t * x = (const uint8_t *)vx;
    const int8_t  * y = (const int8_t  *)vy;

    const int nb         = n / QK_I2_S;
    const int n_tail     = n % QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num      = nb % 32;
    const int groupla_num = la_num != 0 ? 1 : 0;

    int sumi = 0;

#if defined(__AVX2__)
    {
        __m256i mask = _mm256_set1_epi8(0x03);
        __m256i accu = _mm256_setzero_si256();

        for (int i = 0; i < group32_num; i++) {
            __m256i accu32 = _mm256_setzero_si256();
            for (int j = 0; j < 32; j++) {
                __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + i * 32 * 32 + j * 32));
                __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
                __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
                __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

                xq8_3 = _mm256_and_si256(xq8_3, mask);
                xq8_2 = _mm256_and_si256(xq8_2, mask);
                xq8_1 = _mm256_and_si256(xq8_1, mask);
                xq8_0 = _mm256_and_si256(xq8_0, mask);

                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 0));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 96));

                xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_0, xq8_1));
                accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_2, xq8_3));
            }
            accu = _mm256_add_epi32(_mm256_madd_epi16(accu32, _mm256_set1_epi16(1)), accu);
        }

        for (int i = 0; i < groupla_num; i++) {
            __m256i accula = _mm256_setzero_si256();
            for (int j = 0; j < la_num; j++) {
                __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + group32_num * 32 * 32 + j * 32));
                __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
                __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
                __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

                xq8_3 = _mm256_and_si256(xq8_3, mask);
                xq8_2 = _mm256_and_si256(xq8_2, mask);
                xq8_1 = _mm256_and_si256(xq8_1, mask);
                xq8_0 = _mm256_and_si256(xq8_0, mask);

                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 0));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 96));

                xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_0, xq8_1));
                accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_2, xq8_3));
            }
            accu = _mm256_add_epi32(accu, _mm256_madd_epi16(accula, _mm256_set1_epi16(1)));
        }
        sumi = hsum_i32_8(accu);
    }

#elif defined(__ARM_NEON)
    {
        int32x4_t accu_0 = vdupq_n_s32(0);
        int32x4_t accu_1 = vdupq_n_s32(0);
        int32x4_t accu_2 = vdupq_n_s32(0);
        int32x4_t accu_3 = vdupq_n_s32(0);
        const uint8x16_t mask = vdupq_n_u8(3);

        for (int i = 0; i < group32_num; i++) {
#if !defined(__ARM_FEATURE_DOTPROD)
            int16x8_t accu32_0 = vdupq_n_s16(0);
            int16x8_t accu32_1 = vdupq_n_s16(0);
            int16x8_t accu32_2 = vdupq_n_s16(0);
            int16x8_t accu32_3 = vdupq_n_s16(0);
#endif
            for (int j = 0; j < 32; j++) {
                uint8x16_t xq8_6 = vld1q_u8(x + i * 32 * 32 + j * 32);
                uint8x16_t xq8_7 = vld1q_u8(x + i * 32 * 32 + j * 32 + 16);
                uint8x16_t xq8_4 = vshrq_n_u8(xq8_6, 2);
                uint8x16_t xq8_5 = vshrq_n_u8(xq8_7, 2);
                uint8x16_t xq8_2 = vshrq_n_u8(xq8_6, 4);
                uint8x16_t xq8_3 = vshrq_n_u8(xq8_7, 4);
                uint8x16_t xq8_0 = vshrq_n_u8(xq8_6, 6);
                uint8x16_t xq8_1 = vshrq_n_u8(xq8_7, 6);

                int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
                int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
                int8x16_t q8_4 = vreinterpretq_s8_u8(vandq_u8(xq8_4, mask));
                int8x16_t q8_5 = vreinterpretq_s8_u8(vandq_u8(xq8_5, mask));
                int8x16_t q8_6 = vreinterpretq_s8_u8(vandq_u8(xq8_6, mask));
                int8x16_t q8_7 = vreinterpretq_s8_u8(vandq_u8(xq8_7, mask));

                const int8x16_t yq8_0 = vld1q_s8(y + i * 128 * 32 + j * 128 + 0);
                const int8x16_t yq8_1 = vld1q_s8(y + i * 128 * 32 + j * 128 + 16);
                const int8x16_t yq8_2 = vld1q_s8(y + i * 128 * 32 + j * 128 + 32);
                const int8x16_t yq8_3 = vld1q_s8(y + i * 128 * 32 + j * 128 + 48);
                const int8x16_t yq8_4 = vld1q_s8(y + i * 128 * 32 + j * 128 + 64);
                const int8x16_t yq8_5 = vld1q_s8(y + i * 128 * 32 + j * 128 + 80);
                const int8x16_t yq8_6 = vld1q_s8(y + i * 128 * 32 + j * 128 + 96);
                const int8x16_t yq8_7 = vld1q_s8(y + i * 128 * 32 + j * 128 + 112);

#if defined(__ARM_FEATURE_DOTPROD)
                accu_0 = vdotq_s32(accu_0, q8_0, yq8_0);
                accu_1 = vdotq_s32(accu_1, q8_1, yq8_1);
                accu_2 = vdotq_s32(accu_2, q8_2, yq8_2);
                accu_3 = vdotq_s32(accu_3, q8_3, yq8_3);
                accu_0 = vdotq_s32(accu_0, q8_4, yq8_4);
                accu_1 = vdotq_s32(accu_1, q8_5, yq8_5);
                accu_2 = vdotq_s32(accu_2, q8_6, yq8_6);
                accu_3 = vdotq_s32(accu_3, q8_7, yq8_7);
#else
                accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_0), vget_low_s8(yq8_0));
                accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_0), vget_high_s8(yq8_0));
                accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_1), vget_low_s8(yq8_1));
                accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_1), vget_high_s8(yq8_1));
                accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_2), vget_low_s8(yq8_2));
                accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_2), vget_high_s8(yq8_2));
                accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_3), vget_low_s8(yq8_3));
                accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_3), vget_high_s8(yq8_3));
                accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_4), vget_low_s8(yq8_4));
                accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_4), vget_high_s8(yq8_4));
                accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_5), vget_low_s8(yq8_5));
                accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_5), vget_high_s8(yq8_5));
                accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_6), vget_low_s8(yq8_6));
                accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_6), vget_high_s8(yq8_6));
                accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_7), vget_low_s8(yq8_7));
                accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_7), vget_high_s8(yq8_7));
#endif
            }
#if !defined(__ARM_FEATURE_DOTPROD)
            accu_0 = vaddq_s32(accu_0, vmovl_s16(vget_low_s16(accu32_0)));
            accu_0 = vaddq_s32(accu_0, vmovl_high_s16(accu32_0));
            accu_1 = vaddq_s32(accu_1, vmovl_s16(vget_low_s16(accu32_1)));
            accu_1 = vaddq_s32(accu_1, vmovl_high_s16(accu32_1));
            accu_2 = vaddq_s32(accu_2, vmovl_s16(vget_low_s16(accu32_2)));
            accu_2 = vaddq_s32(accu_2, vmovl_high_s16(accu32_2));
            accu_3 = vaddq_s32(accu_3, vmovl_s16(vget_low_s16(accu32_3)));
            accu_3 = vaddq_s32(accu_3, vmovl_high_s16(accu32_3));
#endif
        }

        for (int i = 0; i < groupla_num; i++) {
#if !defined(__ARM_FEATURE_DOTPROD)
            int16x8_t accula_0 = vdupq_n_s16(0);
            int16x8_t accula_1 = vdupq_n_s16(0);
            int16x8_t accula_2 = vdupq_n_s16(0);
            int16x8_t accula_3 = vdupq_n_s16(0);
#endif
            for (int j = 0; j < la_num; j++) {
                uint8x16_t xq8_6 = vld1q_u8(x + group32_num * 32 * 32 + j * 32);
                uint8x16_t xq8_7 = vld1q_u8(x + group32_num * 32 * 32 + j * 32 + 16);
                uint8x16_t xq8_4 = vshrq_n_u8(xq8_6, 2);
                uint8x16_t xq8_5 = vshrq_n_u8(xq8_7, 2);
                uint8x16_t xq8_2 = vshrq_n_u8(xq8_6, 4);
                uint8x16_t xq8_3 = vshrq_n_u8(xq8_7, 4);
                uint8x16_t xq8_0 = vshrq_n_u8(xq8_6, 6);
                uint8x16_t xq8_1 = vshrq_n_u8(xq8_7, 6);

                int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
                int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
                int8x16_t q8_4 = vreinterpretq_s8_u8(vandq_u8(xq8_4, mask));
                int8x16_t q8_5 = vreinterpretq_s8_u8(vandq_u8(xq8_5, mask));
                int8x16_t q8_6 = vreinterpretq_s8_u8(vandq_u8(xq8_6, mask));
                int8x16_t q8_7 = vreinterpretq_s8_u8(vandq_u8(xq8_7, mask));

                const int8x16_t yq8_0 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 0);
                const int8x16_t yq8_1 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 16);
                const int8x16_t yq8_2 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 32);
                const int8x16_t yq8_3 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 48);
                const int8x16_t yq8_4 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 64);
                const int8x16_t yq8_5 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 80);
                const int8x16_t yq8_6 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 96);
                const int8x16_t yq8_7 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 112);

#if defined(__ARM_FEATURE_DOTPROD)
                accu_0 = vdotq_s32(accu_0, q8_0, yq8_0);
                accu_1 = vdotq_s32(accu_1, q8_1, yq8_1);
                accu_2 = vdotq_s32(accu_2, q8_2, yq8_2);
                accu_3 = vdotq_s32(accu_3, q8_3, yq8_3);
                accu_0 = vdotq_s32(accu_0, q8_4, yq8_4);
                accu_1 = vdotq_s32(accu_1, q8_5, yq8_5);
                accu_2 = vdotq_s32(accu_2, q8_6, yq8_6);
                accu_3 = vdotq_s32(accu_3, q8_7, yq8_7);
#else
                accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_0), vget_low_s8(yq8_0));
                accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_0), vget_high_s8(yq8_0));
                accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_1), vget_low_s8(yq8_1));
                accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_1), vget_high_s8(yq8_1));
                accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_2), vget_low_s8(yq8_2));
                accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_2), vget_high_s8(yq8_2));
                accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_3), vget_low_s8(yq8_3));
                accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_3), vget_high_s8(yq8_3));
                accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_4), vget_low_s8(yq8_4));
                accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_4), vget_high_s8(yq8_4));
                accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_5), vget_low_s8(yq8_5));
                accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_5), vget_high_s8(yq8_5));
                accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_6), vget_low_s8(yq8_6));
                accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_6), vget_high_s8(yq8_6));
                accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_7), vget_low_s8(yq8_7));
                accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_7), vget_high_s8(yq8_7));
#endif
            }
#if !defined(__ARM_FEATURE_DOTPROD)
            accu_0 = vaddq_s32(accu_0, vmovl_s16(vget_low_s16(accula_0)));
            accu_0 = vaddq_s32(accu_0, vmovl_high_s16(accula_0));
            accu_1 = vaddq_s32(accu_1, vmovl_s16(vget_low_s16(accula_1)));
            accu_1 = vaddq_s32(accu_1, vmovl_high_s16(accula_1));
            accu_2 = vaddq_s32(accu_2, vmovl_s16(vget_low_s16(accula_2)));
            accu_2 = vaddq_s32(accu_2, vmovl_high_s16(accula_2));
            accu_3 = vaddq_s32(accu_3, vmovl_s16(vget_low_s16(accula_3)));
            accu_3 = vaddq_s32(accu_3, vmovl_high_s16(accula_3));
#endif
        }
        accu_0 = vaddq_s32(accu_0, accu_1);
        accu_2 = vaddq_s32(accu_2, accu_3);
        accu_0 = vaddq_s32(accu_0, accu_2);
        sumi = (int)vaddlvq_s32(accu_0);
    }

#elif defined(__riscv_v_intrinsic) || defined(__riscv_vector)
    sumi = ggml_vec_dot_i2_i8_s_rvv(nb, x, y);

#else
    // Portable scalar fallback
    for (int b = 0; b < nb; ++b) {
        sumi += ggml_i2_s_block_dot_scalar(
            x + b * QK_I2_S_PACKED_BYTES, y + b * QK_I2_S, QK_I2_S);
    }
#endif

    // Tail for n not a multiple of QK_I2_S (applies to every path).
    if (n_tail > 0) {
        sumi += ggml_i2_s_block_dot_scalar(
            x + nb * QK_I2_S_PACKED_BYTES, y + nb * QK_I2_S, n_tail);
    }

    *s = (float)sumi;
}
