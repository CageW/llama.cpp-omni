// mc15: the metal-m5 fork's Metal backend (llama.cpp 049d976bc) built against llama.cpp-omni's ggml (upstream
// cb47092b0 + omni changes). Names the fork's backend uses that this ggml does not have: types and ops that do not
// exist here get values no tensor of this ggml can carry, so their cases are dead code; the two accessors read the
// fields where this ggml stores the same information.
#pragma once

#include "ggml.h"

#define GGML_TYPE_Q2_0            ((enum ggml_type) 60)
#define GGML_TYPE_F8_E4M3         ((enum ggml_type) 61)
#define GGML_GLU_OP_SWIGLU_CLAMP  ((enum ggml_glu_op) 7)
#define GGML_OP_COL2IM_1D         ((enum ggml_op) 120)
#define GGML_OP_LIGHTNING_INDEXER ((enum ggml_op) 121)
#define GGML_OP_TOP_K_MASK        ((enum ggml_op) 122)
#define GGML_OP_DSV4_HC_COMB      ((enum ggml_op) 123)
#define GGML_OP_DSV4_HC_PRE       ((enum ggml_op) 124)
#define GGML_OP_DSV4_HC_POST      ((enum ggml_op) 125)
#define GGML_PREC_BF16            ((enum ggml_prec) 15)

// this ggml keeps the MUL_MAT / MUL_MAT_ID precision in op_params[0] (ggml_mul_mat_set_prec) and has no per-source precision
static inline enum ggml_prec ggml_prec_get_src(const struct ggml_tensor * a, int idx) {
    if ((a->op == GGML_OP_MUL_MAT || a->op == GGML_OP_MUL_MAT_ID) && idx == 1) {
        return (enum ggml_prec) a->op_params[0];
    }
    return GGML_PREC_DEFAULT;
}

// this ggml's ROPE keeps n_ctx (always 0) where the fork keeps no_trunc, so every ROPE here truncates
static inline bool ggml_rope_get_truncate(const struct ggml_tensor * a) {
    return a->op_params[3] == 0;
}
