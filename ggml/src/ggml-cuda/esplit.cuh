#pragma once
// omni: expert-parallel (EP) support for MUL_MAT_ID.
// An "expert-split" buffer type stores a 3-D expert weight tensor [ne0, ne1, n_expert] sharded along the expert
// dimension over a list of devices (each device owns a contiguous range of experts). MUL_MAT_ID on such a tensor
// runs on the context's main device: the (token, expert) columns are sorted by expert and quantized there, every
// owner device gathers its columns over NVLink, runs the grouped GEMM for its own experts and writes the results
// straight into the main device's dst tensor.
#include "common.cuh"

bool ggml_backend_buft_is_cuda_esplit(ggml_backend_buffer_type_t buft);
void ggml_cuda_mul_mat_id_esplit(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
