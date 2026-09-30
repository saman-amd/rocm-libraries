// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_common.cpp -- the AOT plumbing shared by every
 * direct-conv variant.
 *
 * Mirrors the "AOT plumbing shared by every direct-conv variant" section of
 * library/kernels/common/conv_direct_grouped.py:
 *
 *   Python                              C++ (this TU)
 *   ---------------------------------   ------------------------------------
 *   emit_direct_params(b, ...)      rocke_dconv_emit_params
 *   direct_a_descriptor_dynamic(...)    rocke_dconv_a_descriptor_dynamic
 *   direct_d_descriptor_dynamic(...)    rocke_dconv_d_descriptor_dynamic
 *   direct_row_loop_unroll(kh, ...)     rocke_dconv_row_loop_unroll
 */
#include <string.h>

#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"

/* ===================================================================== *
 *  emit_direct_params
 *
 *  Declares the kernarg block in conv_abi order. The order IS the ABI:
 *  kernargs are packed positionally from the launch signature, so a
 *  divergence here does not raise a missing-key error -- it silently shifts
 *  every argument past the first difference.
 * ===================================================================== */
void rocke_dconv_emit_params(rocke_ir_builder_t* b,
                             rocke_dconv_params_t* params,
                             const char* direction)
{
    const rocke_type_t* i32 = rocke_i32();
    const rocke_type_t* f16_global = rocke_ptr_type(b, rocke_f16(), "global");
    const bool is_dgrad = (direction != NULL && strcmp(direction, "dgrad") == 0);

    rocke_param_opts_t ro;
    rocke_param_opts_t wo;
    memset(&ro, 0, sizeof(ro));
    ro.noalias = true;
    ro.noalias_set = true;
    ro.readonly = true;
    ro.readonly_set = true;
    ro.align = 16;
    ro.align_set = true;
    memset(&wo, 0, sizeof(wo));
    wo.noalias = true;
    wo.noalias_set = true;
    wo.writeonly = true;
    wo.writeonly_set = true;
    wo.align = 16;
    wo.align_set = true;

    memset(params, 0, sizeof(*params));
    params->A = rocke_b_param(b, "A", f16_global, &ro);
    params->Bp = rocke_b_param(b, "B", f16_global, &ro);
    params->D = rocke_b_param(b, "D", f16_global, &wo);
    params->A_bytes = rocke_b_param(b, "A_bytes", i32, NULL);
    params->B_bytes = rocke_b_param(b, "B_bytes", i32, NULL);
    params->D_bytes = rocke_b_param(b, "D_bytes", i32, NULL);
    params->p_N = rocke_b_param(b, "p_N", i32, NULL);
    params->p_Hi = rocke_b_param(b, "p_Hi", i32, NULL);
    params->p_Wi = rocke_b_param(b, "p_Wi", i32, NULL);
    params->p_Ho = rocke_b_param(b, "p_Ho", i32, NULL);
    params->p_Wo = rocke_b_param(b, "p_Wo", i32, NULL);
    params->p_groups = rocke_b_param(b, "p_groups", i32, NULL);
    params->p_total_c = rocke_b_param(b, "p_total_c", i32, NULL);
    params->p_total_k = rocke_b_param(b, "p_total_k", i32, NULL);
    if(is_dgrad)
    {
        params->p_A_stride_n = rocke_b_param(b, "p_dY_stride_n", i32, NULL);
        params->p_A_stride_hi = rocke_b_param(b, "p_dY_stride_ho", i32, NULL);
        params->p_A_stride_wi = rocke_b_param(b, "p_dY_stride_wo", i32, NULL);
        params->p_D_stride_n = rocke_b_param(b, "p_dX_stride_n", i32, NULL);
        params->p_D_stride_ho = rocke_b_param(b, "p_dX_stride_hi", i32, NULL);
        params->p_D_stride_wo = rocke_b_param(b, "p_dX_stride_wi", i32, NULL);
    }
    else
    {
        params->p_A_stride_n = rocke_b_param(b, "p_A_stride_n", i32, NULL);
        params->p_A_stride_hi = rocke_b_param(b, "p_A_stride_hi", i32, NULL);
        params->p_A_stride_wi = rocke_b_param(b, "p_A_stride_wi", i32, NULL);
        params->p_D_stride_n = rocke_b_param(b, "p_D_stride_n", i32, NULL);
        params->p_D_stride_ho = rocke_b_param(b, "p_D_stride_ho", i32, NULL);
        params->p_D_stride_wo = rocke_b_param(b, "p_D_stride_wo", i32, NULL);
    }
}

/* ===================================================================== *
 *  direct_a_descriptor_dynamic
 *
 *  A[N, H, W, total_c] with runtime extents. Two embeds fold the
 *  conv-spatial algebra in, exactly as the compile-time version did; only
 *  the bounds and the base strides become kernargs. PAD and stride stay
 *  build-time -- they shape the LDS row and the filter-tap offsets.
 * ===================================================================== */
struct rocke_dynamic_tensor_descriptor*
    rocke_dconv_a_descriptor_dynamic(rocke_ir_builder_t* b,
                                     const rocke_dconv_params_t* params,
                                     int pad,
                                     int stride,
                                     const char* w_upper_0,
                                     const char* w_upper_1)
{
    const char* coord_names[4] = {"n", "h", "w", "c"};
    rocke_value_t* strides[4];
    rocke_dynamic_tensor_descriptor_t* desc;

    strides[0] = params->p_A_stride_n;
    strides[1] = params->p_A_stride_hi;
    strides[2] = params->p_A_stride_wi;
    strides[3] = rocke_b_const_i32(b, 1);

    desc = rocke_tensor_descriptor_naive_dynamic(b, "A_nhwc_direct", coord_names, 4, strides);
    if(!desc)
        return NULL;

    {
        const rocke_transform_t* xforms[2];
        rocke_tensor_descriptor_t* chained;

        /* embed(("y_iter",) -> "h", strides=(1,), offset=-PAD, lo=0, hi=Hi) */
        {
            const char* h_upper[1] = {"y_iter"};
            rocke_value_t* h_strides[1] = {NULL};
            const int h_strides_c[1] = {1};
            xforms[0] = rocke_embed_dynamic_mixed(
                b, h_upper, 1, "h", h_strides, h_strides_c, NULL, -pad, 0, params->p_Hi);
            if(!xforms[0])
                return NULL;
        }
        /* embed(w_upper -> "w", strides=(stride, 1), offset=-PAD, lo=0, hi=Wi) */
        {
            const char* w_upper[2] = {w_upper_0, w_upper_1};
            rocke_value_t* w_strides[2] = {NULL, NULL};
            const int w_strides_c[2] = {stride, 1};
            xforms[1] = rocke_embed_dynamic_mixed(
                b, w_upper, 2, "w", w_strides, w_strides_c, NULL, -pad, 0, params->p_Wi);
            if(!xforms[1])
                return NULL;
        }

        chained = rocke_tensor_descriptor_transform(b, &desc->base, xforms, 2);
        if(!chained)
            return NULL;
        desc->base = *chained;
    }
    return desc;
}

/* ===================================================================== *
 *  direct_d_descriptor_dynamic -- D[N, Ho, Wo, total_k], no transforms.
 * ===================================================================== */
struct rocke_dynamic_tensor_descriptor*
    rocke_dconv_d_descriptor_dynamic(rocke_ir_builder_t* b, const rocke_dconv_params_t* params)
{
    const char* coord_names[4] = {"n", "h", "w", "k"};
    rocke_value_t* strides[4];

    strides[0] = params->p_D_stride_n;
    strides[1] = params->p_D_stride_ho;
    strides[2] = params->p_D_stride_wo;
    strides[3] = rocke_b_const_i32(b, 1);

    return rocke_tensor_descriptor_naive_dynamic(b, "D_nhwk_direct", coord_names, 4, strides);
}

/* ===================================================================== *
 *  direct_row_loop_unroll
 *
 *  The row loop has build-time-periodic pieces that a runtime induction
 *  variable would otherwise break: the accumulator slot (y - r) % KH rotates
 *  with period KH, and where the variant stages rows through LDS the
 *  ping-pong buffer alternates with period 2 -- and an LDS allocation is a
 *  build-time SSA value that cannot be selected by a runtime index.
 *  Unrolling by the lcm lines both up with the body boundary, so every
 *  buffer choice and slot index inside is a constant again while y stays
 *  runtime.
 * ===================================================================== */
int rocke_dconv_row_loop_unroll(int kh, bool lds_ping_pong)
{
    if(!lds_ping_pong)
        return kh;
    return (kh % 2 == 0) ? kh : 2 * kh;
}
