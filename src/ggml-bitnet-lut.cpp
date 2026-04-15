#include <vector>
#include <type_traits>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include "ggml-bitnet.h"
#include "ggml-quants.h"
#include "bitnet-lut-kernels.h"

#if defined(GGML_BITNET_ARM_TL1)

void ggml_bitnet_init(void) {
    // LOG(INFO) << "ggml_bitnet_init";

    if (initialized) {
        return;
    }
    initialized = true;

    // if (wrapper == nullptr) {
    //     wrapper = new BITNET::BITNETGeMMWrapper<bitnet_bitnet_float_type>();
    // }
    if (bitnet_tensor_extras == nullptr) {
        bitnet_tensor_extras = new bitnet_tensor_extra[GGML_BITNET_MAX_NODES];
    }
    bitnet_tensor_extras_index = 0;
}

void ggml_bitnet_free(void) {
    // LOG(INFO) << "ggml_bitnet_free";

    if (!initialized) {
        return;
    }
    initialized = false;

    // delete wrapper;
    // wrapper = nullptr;
    for (size_t i = 0; i < bitnet_tensor_extras_index; i++) {
        // aligned_free(bitnet_tensor_extras[i].qweights);
        // aligned_free(bitnet_tensor_extras[i].scales);
    }
    delete[] bitnet_tensor_extras;
    bitnet_tensor_extras = nullptr;
}

static bool do_permutate(enum ggml_type type) {
    if (type == GGML_TYPE_TL1) {
        // Add additional args to decide if permuted I2 or naive I2
        return false;
    } else {
        return true;
    }
}

bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        src0->backend == GGML_BACKEND_TYPE_CPU) {
        if (src1->ne[1] <= 1) {
            return true;
        }
    }
    return false;
}

size_t ggml_bitnet_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const size_t ne01 = src0->ne[1];
    const size_t ne10 = src1->ne[0];
    const size_t ne11 = src1->ne[1];
    const int bits = ggml_bitnet_get_type_bits(src0->type);
    
    size_t wsize = ne10 * ne11 * 15 * sizeof(int8_t) + 1 * ne11 * 2 * sizeof(bitnet_float_type);
    if (sizeof(bitnet_float_type) == 2) {
        // Need fp32 to fp16 conversion
        wsize += std::max(ne10, ne01) * ne11 * sizeof(bitnet_float_type);
    }
    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}

int ggml_bitnet_get_type_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_TL1:
            return 2;
        case GGML_TYPE_Q4_0:
            return 4;
        default:
            return 0;
    }
}

#endif
#if defined(GGML_BITNET_X86_TL2)
void ggml_bitnet_init(void) {
    // LOG(INFO) << "ggml_bitnet_init";

    if (initialized) {
        return;
    }
    initialized = true;

    // if (wrapper == nullptr) {
    //     wrapper = new BITNET::BITNETGeMMWrapper<bitnet_bitnet_float_type>();
    // }
    if (bitnet_tensor_extras == nullptr) {
        bitnet_tensor_extras = new bitnet_tensor_extra[GGML_BITNET_MAX_NODES];
    }
    bitnet_tensor_extras_index = 0;
}

void ggml_bitnet_free(void) {
    // LOG(INFO) << "ggml_bitnet_free";

    if (!initialized) {
        return;
    }
    initialized = false;

    // delete wrapper;
    // wrapper = nullptr;
    for (size_t i = 0; i < bitnet_tensor_extras_index; i++) {
        // aligned_free(bitnet_tensor_extras[i].qweights);
        // aligned_free(bitnet_tensor_extras[i].scales);
    }
    delete[] bitnet_tensor_extras;
    bitnet_tensor_extras = nullptr;
}

bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        src0->backend == GGML_BACKEND_TYPE_CPU) {
        return true;
    }
    return false;
}

size_t ggml_bitnet_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const size_t ne01 = src0->ne[1];
    const size_t ne10 = src1->ne[0];
    const size_t ne11 = src1->ne[1];
    
    size_t wsize = ne10 * ne11 * 11 * sizeof(int8_t) + 2 * ne11 * 2 * sizeof(bitnet_float_type);
    if (sizeof(bitnet_float_type) == 2) {
        // Need fp32 to fp16 conversion
        wsize += std::max(ne10, ne01) * ne11 * sizeof(bitnet_float_type);
    }
    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}

int ggml_bitnet_get_type_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_TL2:
            return 2;
        case GGML_TYPE_Q4_0:
            return 4;
        default:
            return 0;
    }
}
#endif

// ============================================================================
// RISC-V TL3: real table-lookup path for BitNet b1.58 weights on RVV.
// ============================================================================
#if defined(GGML_BITNET_RISCV_TL3)

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
    for (size_t i = 0; i < bitnet_tensor_extras_index; ++i) {
        aligned_free(bitnet_tensor_extras[i].qweights);
        aligned_free(bitnet_tensor_extras[i].scales);
    }
    delete[] bitnet_tensor_extras;
    bitnet_tensor_extras = nullptr;
    bitnet_tensor_extras_index = 0;
}

bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0,
                             const struct ggml_tensor * src1,
                             const struct ggml_tensor * dst) {
    if (!is_type_supported(src0->type)) return false;
    if (src1->type != GGML_TYPE_F32)     return false;
    if (dst->type  != GGML_TYPE_F32)     return false;
    if (src0->backend != GGML_BACKEND_TYPE_CPU) return false;
    // GEMV only — batch matmul still falls back to MAD path.
    if (src1->ne[1] > 1) return false;
    // Need K divisible by 4 (group size) and 128 (I2_S block).
    if ((src0->ne[0] % TL3_G)        != 0) return false;
    if ((src0->ne[0] % TL3_QK_I2_S)  != 0) return false;
    return true;
}

size_t ggml_bitnet_mul_mat_get_wsize(const struct ggml_tensor * src0,
                                     const struct ggml_tensor * src1,
                                     const struct ggml_tensor * dst) {
    (void) src1; (void) dst;
    const size_t k        = (size_t) src0->ne[0];
    const size_t n_groups = k / TL3_G;
    // qlut int16 table + one fp32 lut scale
    size_t wsize = n_groups * TL3_LUT_BYTES + 2 * sizeof(bitnet_float_type);
    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}

int ggml_bitnet_get_type_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_TL1:
        case GGML_TYPE_I2_S: return 2;
        case GGML_TYPE_Q4_0: return 4;
        default:             return 0;
    }
}

// Top-level hooks exposed to ggml.c (C linkage via extern "C" in ggml-bitnet.h)
void ggml_preprocessor(int m, int k, void * B, void * LUT_Scales, void * QLUT) {
    ggml_preprocessor_impl(m, k, B, LUT_Scales, QLUT);
}

void ggml_qgemm_lut(int m, int k, void * A, void * LUT,
                    void * Scales, void * LUT_Scales, void * C) {
    ggml_qgemm_lut_impl(m, k, A, LUT, Scales, LUT_Scales, C);
}

void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor) {
    ggml_bitnet_transform_tensor_impl(tensor);
}

// Stubs for APIs declared in ggml-bitnet.h but unused by TL3.
void ggml_bitnet_mul_mat_task_init(void *, void *, void *, void *, int, int, int, int) {}
void ggml_bitnet_mul_mat_task_compute(void *, void *, void *, void *, void *, void *, int, int, int, int) {}
void ggml_bitnet_set_n_threads(int) {}

#else
// ----------------------------------------------------------------------------
// Fallback when TL3 is not enabled: no-op hooks so the rest of the CPU path
// (ggml-bitnet-mad.cpp vec_dot_i2_i8_s) is free to handle everything.
// ----------------------------------------------------------------------------

void ggml_bitnet_init(void) {}
void ggml_bitnet_free(void) {}

bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0,
                             const struct ggml_tensor * src1,
                             const struct ggml_tensor * dst) {
    (void) src0; (void) src1; (void) dst;
    return false;
}

size_t ggml_bitnet_mul_mat_get_wsize(const struct ggml_tensor * src0,
                                     const struct ggml_tensor * src1,
                                     const struct ggml_tensor * dst) {
    (void) src0; (void) src1; (void) dst;
    return 0;
}

int ggml_bitnet_get_type_bits(enum ggml_type type) {
    (void) type;
    return 0;
}

#endif // GGML_BITNET_RISCV_TL3
