#include <vector>
#include <type_traits>

#include "ggml-bitnet.h"
#include "ggml-quants.h"
#include "bitnet-lut-kernels.h"

#if defined(GGML_BITNET_ARM_TL1)

void ggml_bitnet_init() {
    if (initialized) return;
    initialized = true;

    if (!bitnet_tensor_extras)
        bitnet_tensor_extras = new bitnet_tensor_extra[GGML_BITNET_MAX_NODES];

    bitnet_tensor_extras_index = 0;
}

void ggml_bitnet_free() {
    if (!initialized) return;
    initialized = false;

    delete[] bitnet_tensor_extras;
    bitnet_tensor_extras = nullptr;
}

static bool do_permutate(ggml_type type) {
    return type != GGML_TYPE_TL1;
}

bool ggml_bitnet_can_mul_mat(const ggml_tensor* src0, const ggml_tensor* src1, const ggml_tensor* dst) {
    return is_type_supported(src0->type) && 
           src1->type == GGML_TYPE_F32 &&
           dst->type == GGML_TYPE_F32 && 
           src0->backend == GGML_BACKEND_TYPE_CPU &&
           src1->ne[1] <= 1;
}

size_t ggml_bitnet_mul_mat_get_wsize(const ggml_tensor* src0, const ggml_tensor* src1, const ggml_tensor* dst) {
    size_t wsize = src1->ne[0] * src1->ne[1] * 15 * sizeof(int8_t) + src1->ne[1] * 2 * sizeof(bitnet_float_type);

    if (sizeof(bitnet_float_type) == 2)
        wsize += std::max(src0->ne[1], src1->ne[0]) * src1->ne[1] * sizeof(bitnet_float_type);

    return ((wsize - 1) / 64 + 1) * 64;
}

int ggml_bitnet_get_type_bits(ggml_type type) {
    return (type == GGML_TYPE_TL1) ? 2 : (type == GGML_TYPE_Q4_0) ? 4 : 0;
}

#endif

#if defined(GGML_BITNET_X86_TL2)

void ggml_bitnet_init() {
    if (initialized) return;
    initialized = true;

    if (!bitnet_tensor_extras)
        bitnet_tensor_extras = new bitnet_tensor_extra[GGML_BITNET_MAX_NODES];

    bitnet_tensor_extras_index = 0;
}

void ggml_bitnet_free() {
    if (!initialized) return;
    initialized = false;

    delete[] bitnet_tensor_extras;
    bitnet_tensor_extras = nullptr;
}

bool ggml_bitnet_can_mul_mat(const ggml_tensor* src0, const ggml_tensor* src1, const ggml_tensor* dst) {
    return is_type_supported(src0->type) &&
           src1->type == GGML_TYPE_F32 &&
           dst->type == GGML_TYPE_F32 &&
           src0->backend == GGML_BACKEND_TYPE_CPU;
}

size_t ggml_bitnet_mul_mat_get_wsize(const ggml_tensor* src0, const ggml_tensor* src1, const ggml_tensor* dst) {
    size_t wsize = src1->ne[0] * src1->ne[1] * 11 * sizeof(int8_t) + src1->ne[1] * 2 * sizeof(bitnet_float_type);

    if (sizeof(bitnet_float_type) == 2)
        wsize += std::max(src0->ne[1], src1->ne[0]) * src1->ne[1] * sizeof(bitnet_float_type);

    return ((wsize - 1) / 64 + 1) * 64;
}

int ggml_bitnet_get_type_bits(ggml_type type) {
    return (type == GGML_TYPE_TL2) ? 2 : (type == GGML_TYPE_Q4_0) ? 4 : 0;
}
// fix update

#endif
