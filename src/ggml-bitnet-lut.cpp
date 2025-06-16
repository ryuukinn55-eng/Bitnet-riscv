#include <vector>
#include <type_traits>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include "ggml-bitnet.h"
#include "ggml-quants.h"
#include "bitnet-lut-kernels.h"

#if defined(__riscv) || defined(__riscv__)

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

static bool initialized = false;
static bitnet_tensor_extra* bitnet_tensor_extras = nullptr;
static size_t bitnet_tensor_extras_index = 0;

#ifndef GGML_BITNET_MAX_NODES
#define GGML_BITNET_MAX_NODES 8192
#endif

#ifndef bitnet_float_type
typedef float bitnet_float_type;
#endif

static bool is_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_TL1:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            return true;
        default:
            return false;
    }
}

void ggml_bitnet_init(void) {
    if (initialized) return;
    initialized = true;

    if (bitnet_tensor_extras == nullptr) {
        bitnet_tensor_extras = new bitnet_tensor_extra[GGML_BITNET_MAX_NODES];
    }
    bitnet_tensor_extras_index = 0;
}

void ggml_bitnet_free(void) {
    if (!initialized) return;
    initialized = false;

    for (size_t i = 0; i < bitnet_tensor_extras_index; i++) {
        // Free any buffers if needed
    }

    delete[] bitnet_tensor_extras;
    bitnet_tensor_extras = nullptr;
}

static bool do_permutate(enum ggml_type type) {
    return type != GGML_TYPE_TL1;
}

bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    bool is_cpu_accessible = true;
    return is_type_supported(src0->type) &&
           src1->type == GGML_TYPE_F32 &&
           dst->type == GGML_TYPE_F32 &&
           is_cpu_accessible &&
           src1->ne[1] <= 1;
}

size_t ggml_bitnet_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const size_t ne01 = src0->ne[1];
    const size_t ne10 = src1->ne[0];
    const size_t ne11 = src1->ne[1];
    const int bits = ggml_bitnet_get_type_bits(src0->type);

    size_t wsize = ne10 * ne11 * 15 * sizeof(int8_t) + 1 * ne11 * 2 * sizeof(bitnet_float_type);
    if (sizeof(bitnet_float_type) == 2) {
        wsize += std::max(ne10, ne01) * ne11 * sizeof(bitnet_float_type);
    }

    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}

int ggml_bitnet_get_type_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_TL1: return 2;
        case GGML_TYPE_Q4_0: return 4;
        default: return 0;
    }
}

void ggml_bitnet_mul_mat(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const void * wdata,
    size_t wsize) {

    const float * B = (const float *)src1->data;
    float * C = (float *)dst->data;
    const float * A = (const float *)src0->data;

    const int M = src0->ne[1];
    const int K = src0->ne[0];
    const int N = src1->ne[0];

    // RVV-accelerated matmul
#if defined(__riscv_vector)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            int k = 0;
            size_t vl;
            for (; k < K;) {
                vl = vsetvl_e32m1(K - k);
                vfloat32m1_t a = vle32_v_f32m1(&A[k + m * K], vl);
                vfloat32m1_t b = vle32_v_f32m1(&B[k + n * K], vl);
                vfloat32m1_t prod = vfmul_vv_f32m1(a, b, vl);
                sum += vfredusum_vs_f32m1_f32m1(prod, vfmv_s_f_f32m1(prod, 0.0f), vl)[0];
                k += vl;
            }
            C[n + m * N] = sum;
        }
    }
#else
    // Scalar fallback
    for (int i = 0; i < M * N; ++i) {
        C[i] = 0.0f;
    }

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = A[k + m * K];
                float b = B[k + n * K];
                sum += a * b;
            }
            C[n + m * N] = sum;
        }
    }
#endif
}

#endif // __riscv