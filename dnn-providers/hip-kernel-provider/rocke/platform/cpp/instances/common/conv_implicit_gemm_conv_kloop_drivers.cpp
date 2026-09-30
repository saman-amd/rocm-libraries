// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_implicit_gemm_conv_kloop_drivers.c -- C99 port of the three
 * K-loop drivers of build_implicit_gemm_conv
 * (rocke/instances/common/conv_implicit_gemm.py, lines 1276-1347).
 *
 * Each driver sequences the per-K-tile load + compute phases and writes the
 * final-tile accumulators into ctx->final_accs / ctx->num_final_accs, which the
 * epilogue phase then reads. Exactly one driver is called per build, chosen by
 * the public driver as Python does:
 *   spec.unroll_k          -> rocke_conv_emit_kloop_unroll
 *   else not async_dma     -> rocke_conv_emit_kloop_simple
 *   else (async_dma)       -> rocke_conv_emit_kloop_async
 *
 * The builder-call sequence here is byte-identical to the Python source span.
 * Phase functions (emit_load_phase / emit_mfma_phase) are peers reached through
 * the internal header; this TU touches only ctx + the builder it carries.
 */
#include "rocke/instance_conv_implicit_gemm_internal.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* ----- shared small helper: copy a working acc array into ctx->final_accs ----
 * Python sets `final_accs = current_accs` (or `for_op.results`); in C the
 * drivers write the ctx slot the epilogue reads. */
/* Every driver now walks a runtime [k_lo, k_hi) range with an scf.for, so the
 * old build-time tile-offset helper (const_i32(it*block_k), plus the slice base
 * for wgrad) has no callers left: the loop induction variable *is* the offset.
 */

static void
    rocke_conv_set_final_accs(rocke_conv_build_ctx_t* ctx, rocke_value_t* const* accs, int num_accs)
{
    int i;
    ctx->num_final_accs = num_accs;
    for(i = 0; i < num_accs; ++i)
        ctx->final_accs[i] = accs[i];
}

/* ===================================================================== *
 * rocke_conv_emit_kloop_unroll   (Python lines 1276-1310)
 *
 * Double-buffered Python-unrolled K-loop software pipeline (ping-pong
 * A_smem/A_smem2). Stage tile it+1 into the alternate LDS buffer while the MFMA
 * for tile it reads the current buffer. One barrier per iteration publishes the
 * prefetched tile and orders the current tile's ds_reads ahead of the it+2
 * prefetch that reuses the same buffer two iterations later.
 * ===================================================================== */
void rocke_conv_emit_kloop_unroll(rocke_conv_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int block_k = ctx->block_k;
    int num_accs = ctx->num_accs;
    int i;

    rocke_iter_arg_t iter_args[ROCKE_CONV_MAX_ACCS];
    rocke_for_t for_op;
    rocke_value_t* iter_vars[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* accs_a[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* accs_b[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* c_2block_k;
    rocke_value_t* k_lo;
    rocke_value_t* k_hi;

    k_lo = (ctx->kloop_k_lo != NULL) ? ctx->kloop_k_lo : ctx->c0;
    k_hi = (ctx->kloop_k_hi != NULL) ? ctx->kloop_k_hi : ctx->c_K_gemm;
    c_2block_k = rocke_b_const_i32(b, 2 * block_k);

    /* Prologue: stage tile 0 into buf0 and publish it. */
    rocke_conv_emit_load_phase(ctx, k_lo, ctx->A_smem, ctx->B_smem);
    rocke_b_sync(b);

    for(i = 0; i < num_accs; ++i)
    {
        iter_args[i].name = ctx->acc_names[i];
        iter_args[i].init = ctx->acc_inits[i];
    }

    /* AOT: the trip count is runtime, so bufs[it % 2] cannot be evaluated --
     * an LDS allocation is a build-time value. The body is unrolled twice and
     * steps by 2*block_k instead, which binds each phase to a build-time
     * buffer while still alternating them. An odd tile count needs no guard:
     * the trailing phase addresses k >= K, whose coords fall outside the
     * descriptor's padded bounds, so the buffer resource returns zero. */
    for_op = rocke_b_scf_for_iter(b,
                                  k_lo,
                                  k_hi,
                                  c_2block_k,
                                  iter_args,
                                  num_accs,
                                  "k_unroll",
                                  /*unroll=*/false,
                                  /*elide_trailing_barrier=*/true);
    for(i = 0; i < for_op.num_iter_vars; ++i)
        iter_vars[i] = for_op.iter_vars[i];

    rocke_b_region_enter(b, for_op.body);
    {
        rocke_value_t* k_odd = rocke_b_add(b, for_op.iv, ctx->c_block_k);
        rocke_value_t* k_nxt_pair = rocke_b_add(b, for_op.iv, c_2block_k);

        /* Phase A: prefetch tile k+1 into buf1, MFMA tile k out of buf0. */
        rocke_conv_emit_load_phase(ctx, k_odd, ctx->A_smem2, ctx->B_smem2);
        ctx->k_off_capture = for_op.iv;
        rocke_conv_emit_mfma_phase(
            ctx, ctx->A_smem, ctx->B_smem, iter_vars, for_op.num_iter_vars, accs_a);
        /* Publishes buf1 and drains buf0's ds_reads. */
        rocke_b_sync(b);

        /* Phase B: the buffers swap roles. */
        rocke_conv_emit_load_phase(ctx, k_nxt_pair, ctx->A_smem, ctx->B_smem);
        ctx->k_off_capture = k_odd;
        rocke_conv_emit_mfma_phase(
            ctx, ctx->A_smem2, ctx->B_smem2, accs_a, for_op.num_iter_vars, accs_b);
        rocke_b_sync(b);

        rocke_b_scf_yield(b, accs_b, for_op.num_iter_vars);
    }
    rocke_b_region_leave(b);

    rocke_conv_set_final_accs(ctx, for_op.op->results, for_op.op->num_results);
}

/* ===================================================================== *
 * rocke_conv_emit_kloop_basic
 *
 * CK pipeline_basic: single-buffer, global-read/compute overlap.
 * Mirrors the Python ``elif spec.pipeline == "basic":`` branch.
 *
 * Per-iteration order (byte-identical to Python):
 *   emit_global_read(it+1)   buffer_load_vN (VMEM, in flight)
 *   sync()                   s_waitcnt(lgkmcnt=0) + s_barrier
 *                            (drains prior ds_write; tile it RAW-safe)
 *   k_off_capture = it       descriptor uses tile it's offset for mfma
 *   emit_mfma_phase          ds_read(A_smem,B_smem) + mfma
 *   sync()                   s_waitcnt(lgkmcnt=0) + s_barrier
 *                            (drains ds_reads; A_smem WAR-safe)
 *   emit_lds_write(staged)   smem_store_vN (safe to write now)
 * ===================================================================== */
void rocke_conv_emit_kloop_basic(rocke_conv_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int num_accs = ctx->num_accs;
    int i;

    rocke_iter_arg_t iter_args[ROCKE_CONV_MAX_ACCS];
    rocke_for_t for_op;
    rocke_value_t* k_basic;
    rocke_value_t* iter_vars[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* new_accs[ROCKE_CONV_MAX_ACCS];
    rocke_ctl_staged_t staged_a0;
    rocke_ctl_staged_t staged_b0;
    rocke_ctl_staged_t staged_a_next;
    rocke_ctl_staged_t staged_b_next;
    rocke_value_t* k0_val;
    rocke_value_t* k_lo;
    rocke_value_t* k_hi;

    /* Prologue: fetch tile 0 and write it straight to LDS -- there are no
     * prior ds_reads to drain, so the write can follow immediately.
     * wgrad slices the reduction, so the prologue tile starts at kloop_k_lo. */
    k_lo = (ctx->kloop_k_lo != NULL) ? ctx->kloop_k_lo : ctx->c0;
    k_hi = (ctx->kloop_k_hi != NULL) ? ctx->kloop_k_hi : ctx->c_K_gemm;
    k0_val = k_lo;
    rocke_conv_emit_global_read(ctx, k0_val, &staged_a0, &staged_b0);
    rocke_conv_emit_lds_write(ctx, k0_val, &staged_a0, &staged_b0, ctx->A_smem, ctx->B_smem);

    for(i = 0; i < num_accs; ++i)
    {
        iter_args[i].name = ctx->acc_names[i];
        iter_args[i].init = ctx->acc_inits[i];
    }

    for_op = rocke_b_scf_for_iter(b,
                                  k_lo,
                                  k_hi,
                                  ctx->c_block_k,
                                  iter_args,
                                  num_accs,
                                  "k_basic",
                                  /*unroll=*/false,
                                  /*elide_trailing_barrier=*/true);
    k_basic = for_op.iv;
    for(i = 0; i < for_op.num_iter_vars; ++i)
        iter_vars[i] = for_op.iter_vars[i];

    rocke_b_region_enter(b, for_op.body);
    {
        rocke_value_t* k_next = rocke_b_add(b, k_basic, ctx->c_block_k);
        /* Issue the buffer_load for tile k+1 BEFORE the sync so the VMEM
         * latency overlaps the mfma stream below. Tiles past the reduction
         * extent zero-fill through the descriptor bounds, so no guard. */
        rocke_conv_emit_global_read(ctx, k_next, &staged_a_next, &staged_b_next);
        /* Drain the previous iteration's ds_write, then barrier. */
        rocke_b_sync(b);
        ctx->k_off_capture = k_basic;
        rocke_conv_emit_mfma_phase(
            ctx, ctx->A_smem, ctx->B_smem, iter_vars, for_op.num_iter_vars, new_accs);
        /* Drain ds_reads before the staged tile may overwrite A/B_smem. */
        rocke_b_sync(b);
        rocke_conv_emit_lds_write(
            ctx, k_next, &staged_a_next, &staged_b_next, ctx->A_smem, ctx->B_smem);
        rocke_b_scf_yield(b, new_accs, for_op.num_iter_vars);
    }
    rocke_b_region_leave(b);

    rocke_conv_set_final_accs(ctx, for_op.op->results, for_op.op->num_results);
}

/* ===================================================================== *
 * rocke_conv_emit_kloop_simple   (Python lines 1311-1319)
 *
 * Single scf.for_iter load + sync + mfma + sync. The not-async, not-unroll
 * branch.
 * ===================================================================== */
void rocke_conv_emit_kloop_simple(rocke_conv_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int num_accs = ctx->num_accs;
    int i;

    /* for_op = b.scf_for_iter(c0, c_K_gemm, c_block_k, accs, iv_name="k0") */
    rocke_iter_arg_t iter_args[ROCKE_CONV_MAX_ACCS];
    rocke_for_t for_op;
    rocke_value_t* k0;
    rocke_value_t* iter_vars[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* new_accs[ROCKE_CONV_MAX_ACCS];

    for(i = 0; i < num_accs; ++i)
    {
        iter_args[i].name = ctx->acc_names[i];
        iter_args[i].init = ctx->acc_inits[i];
    }

    /* Start at the slice base, not at a bare zero: wgrad hands the drivers a
     * runtime k_lo and only the forward conv leaves it NULL. Reading ctx->c0
     * here worked while wgrad aliased the two, and silently restarted every
     * split-K slice from 0 once they came apart. */
    for_op = rocke_b_scf_for_iter(b,
                                  (ctx->kloop_k_lo != NULL) ? ctx->kloop_k_lo : ctx->c0,
                                  ctx->c_K_gemm,
                                  ctx->c_block_k,
                                  iter_args,
                                  num_accs,
                                  "k0",
                                  /*unroll=*/false,
                                  /*elide_trailing_barrier=*/true);

    /* with for_op as (k0, iter_vars): */
    k0 = for_op.iv;
    for(i = 0; i < for_op.num_iter_vars; ++i)
        iter_vars[i] = for_op.iter_vars[i];

    rocke_b_region_enter(b, for_op.body);
    {
        /* emit_load_phase(k0, A_smem, B_smem) */
        rocke_conv_emit_load_phase(ctx, k0, ctx->A_smem, ctx->B_smem);
        rocke_b_sync(b);
        /* new_accs = emit_mfma_phase(A_smem, B_smem, iter_vars) */
        rocke_conv_emit_mfma_phase(
            ctx, ctx->A_smem, ctx->B_smem, iter_vars, for_op.num_iter_vars, new_accs);
        rocke_b_sync(b);
        /* b.scf_yield(*new_accs) */
        rocke_b_scf_yield(b, new_accs, for_op.num_iter_vars);
    }
    rocke_b_region_leave(b);

    /* final_accs = for_op.results */
    rocke_conv_set_final_accs(ctx, for_op.op->results, for_op.op->num_results);
}

/* ===================================================================== *
 * rocke_conv_emit_kloop_async   (Python lines 1320-1347)
 *
 * async_dma path: SoftwarePipeline.run_ping_pong over the AsyncTileLoader path.
 * SoftwarePipeline is a peer port; until it lands, this driver reproduces the
 * run_ping_pong sequencing inline (helpers/pipeline.py, run_ping_pong) for the
 * exact policy the conv builder constructs:
 *
 *   SoftwarePipeline(num_iters=K_iters, double_buffer=double_buffer,
 *                    wait_vmcnt=True, sync_after_wait=True,
 *                    sync_before_issue=True, overlap_vmcnt=True)
 *   issue_load(it, buf)      = emit_load_phase(const_i32(it*block_k), buf[0], buf[1])
 *   compute(it, buf, state)  = emit_mfma_phase(buf[0], buf[1], state)
 *   buffers = [(A_smem,B_smem),(A_smem2,B_smem2)]
 *   schedule = ctx->schedule
 *
 * Constructed flags: wait_vmcnt=True, sync_after_wait=True,
 * sync_before_issue=True, overlap_vmcnt=True; num_buffers unset (0) =>
 * nb derived from double_buffer. The two buffer pairs supplied bound nb<=2.
 * ===================================================================== */
void rocke_conv_emit_kloop_async(rocke_conv_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int block_k = ctx->block_k;
    int num_accs = ctx->num_accs;
    int i;

    /* Pipeline flags, matching the SoftwarePipeline the conv builder
     * constructs: wait_vmcnt, sync_after_wait, sync_before_issue and
     * overlap_vmcnt all true, two buffers. */
    const bool wait_vmcnt = true;
    const bool sync_after_wait = true;
    const bool sync_before_issue = true;
    const bool overlap_vmcnt = true;

    rocke_iter_arg_t iter_args[ROCKE_CONV_MAX_ACCS];
    rocke_for_t for_op;
    rocke_value_t* iter_vars[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* accs_a[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* accs_b[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* c_bk;
    rocke_value_t* c_2bk;
    rocke_value_t* k_lo;
    rocke_value_t* k_hi;

    k_lo = (ctx->kloop_k_lo != NULL) ? ctx->kloop_k_lo : ctx->c0;
    k_hi = (ctx->kloop_k_hi != NULL) ? ctx->kloop_k_hi : ctx->c_K_gemm;
    c_bk = rocke_b_const_i32(b, block_k);
    c_2bk = rocke_b_const_i32(b, 2 * block_k);

    /* Prologue: stage tile 0 into buf0. */
    rocke_conv_emit_load_phase(ctx, k_lo, ctx->A_smem, ctx->B_smem);

    for(i = 0; i < num_accs; ++i)
    {
        iter_args[i].name = ctx->acc_names[i];
        iter_args[i].init = ctx->acc_inits[i];
    }

    /* Mirrors SoftwarePipeline.run_ping_pong_dynamic: the body is unrolled
     * twice and steps by 2*block_k so each phase binds a build-time LDS
     * buffer while still alternating them. */
    for_op = rocke_b_scf_for_iter(b,
                                  k_lo,
                                  k_hi,
                                  c_2bk,
                                  iter_args,
                                  num_accs,
                                  "k_pipe",
                                  /*unroll=*/false,
                                  /*elide_trailing_barrier=*/true);
    for(i = 0; i < for_op.num_iter_vars; ++i)
        iter_vars[i] = for_op.iter_vars[i];

    rocke_b_region_enter(b, for_op.body);
    {
        rocke_value_t* k1 = rocke_b_add(b, for_op.iv, c_bk);
        rocke_value_t* k2 = rocke_b_add(b, for_op.iv, c_2bk);

        /* ---- phase A: compute tile k from buf0, stream k+1 into buf1 ---- */
        if(sync_before_issue)
        {
            if(overlap_vmcnt)
                rocke_b_sync_lds_only(b);
            else
                rocke_b_sync(b);
        }
        rocke_conv_emit_load_phase(ctx, k1, ctx->A_smem2, ctx->B_smem2);
        if(wait_vmcnt)
        {
            rocke_b_s_waitcnt(b, overlap_vmcnt ? 1 : 0, -1, -1);
        }
        if(sync_after_wait)
        {
            if(overlap_vmcnt)
                rocke_b_sync_lds_only(b);
            else
                rocke_b_sync(b);
        }
        rocke_schedule_policy_emit_compute_prologue(&ctx->schedule, b);
        ctx->k_off_capture = for_op.iv;
        rocke_conv_emit_mfma_phase(
            ctx, ctx->A_smem, ctx->B_smem, iter_vars, for_op.num_iter_vars, accs_a);
        rocke_schedule_policy_emit_compute_epilogue(&ctx->schedule, b);

        /* ---- phase B: the buffers swap roles ---- */
        if(sync_before_issue)
        {
            if(overlap_vmcnt)
                rocke_b_sync_lds_only(b);
            else
                rocke_b_sync(b);
        }
        rocke_conv_emit_load_phase(ctx, k2, ctx->A_smem, ctx->B_smem);
        if(wait_vmcnt)
        {
            rocke_b_s_waitcnt(b, overlap_vmcnt ? 1 : 0, -1, -1);
        }
        if(sync_after_wait)
        {
            if(overlap_vmcnt)
                rocke_b_sync_lds_only(b);
            else
                rocke_b_sync(b);
        }
        rocke_schedule_policy_emit_compute_prologue(&ctx->schedule, b);
        ctx->k_off_capture = k1;
        rocke_conv_emit_mfma_phase(
            ctx, ctx->A_smem2, ctx->B_smem2, accs_a, for_op.num_iter_vars, accs_b);
        rocke_schedule_policy_emit_compute_epilogue(&ctx->schedule, b);

        rocke_b_scf_yield(b, accs_b, for_op.num_iter_vars);
    }
    rocke_b_region_leave(b);

    /* The final phase left a prefetch in flight behind an LDS-scoped barrier.
     * Drain both before the epilogue, which stages through the same LDS. */
    rocke_b_s_waitcnt(b, 0, -1, -1);
    rocke_b_sync(b);

    rocke_conv_set_final_accs(ctx, for_op.op->results, for_op.op->num_results);
}

/* ===================================================================== *
 * rocke_conv_emit_kloop_wavelet   (Python lines 1539-1797)
 *
 * Dedicated load-wave / math-wave split pipeline (CK Tile #8009).
 *
 * The workgroup is split into two roles identified by warp_id:
 *   math waves  [0,         n_math_warps)  -- LDS reads + WMMA/MFMA + epilogue
 *   load waves  [n_math_warps, total_warps) -- DRAM→register fetch + LDS write
 *
 * Two sub-paths:
 *
 *   WMMA path (ctx->is_wmma == true, e.g. gfx1250):
 *     Uses scf_if_else (br i1) with a shared join block.  Both branches emit
 *     s_barrier calls; the shared join prevents simplifycfg from removing them.
 *
 *   MFMA path (ctx->is_wmma == false, e.g. gfx942):
 *     Uses exec-mask instructions (exec_and_saveexec / exec_xor /
 *     exec_or_saveexec / exec_or) to flat-sequentially encode the split.
 *     LLVM divergent branches would deadlock the barriers; exec-mask avoids
 *     that by keeping all threads in one basic block while suppressing VGPR
 *     writes for the inactive side.
 *
 * Barrier protocol (must be bit-identical in both branches):
 *
 *   MATH branch:
 *     barrier_0
 *     for i in 0..K_iters-2:
 *       MFMA(LDS)
 *       barrier_A   <- math done reading LDS
 *       barrier_B   <- wait for load to write next tile
 *     MFMA(LDS)     <- tail, no barriers
 *     [epilogue -- emits epi_barriers barriers]
 *
 *   LOAD branch:
 *     fetch tile 0 -> regs
 *     store regs -> LDS
 *     barrier_0
 *     for i in 0..K_iters-2:
 *       fetch tile i+1 -> regs    <- overlaps math MFMA
 *       barrier_A                 <- wait for math to release LDS
 *       store regs -> LDS
 *       barrier_B                 <- signal LDS ready
 *     [epilogue stub -- epi_barriers bare barriers, no stores]
 * ===================================================================== */

/* wavelet_fetch: issue buffer_load_vN for A and B into VGPR staging using
 * load_tid (load-wave-relative thread index). Sets k_off_capture.
 * Mirrors Python wavelet_fetch = a_wavelet_loader.fetch(b, k_off, load_tid). */
static void wavelet_fetch(rocke_conv_build_ctx_t* ctx,
                          rocke_value_t* k_off,
                          rocke_ctl_staged_t* a_staged,
                          rocke_ctl_staged_t* b_staged)
{
    rocke_ir_builder_t* b = ctx->b;
    ctx->k_off_capture = k_off;
    rocke_coalesced_tile_loader_load_global(b,
                                            &ctx->a_wavelet_loader,
                                            ctx->wavelet_load_tid,
                                            rocke_conv_a_descriptor,
                                            ctx,
                                            ctx->a_rsrc,
                                            NULL,
                                            a_staged);
    rocke_coalesced_tile_loader_load_global(b,
                                            &ctx->b_wavelet_loader,
                                            ctx->wavelet_load_tid,
                                            rocke_conv_b_descriptor,
                                            ctx,
                                            ctx->b_rsrc,
                                            NULL,
                                            b_staged);
}

/* wavelet_store: write staged VGPRs to LDS.
 * Mirrors Python wavelet_store = a_wavelet_loader.store_fetched(b, a_regs, A_smem). */
static void wavelet_store(rocke_conv_build_ctx_t* ctx,
                          const rocke_ctl_staged_t* a_staged,
                          const rocke_ctl_staged_t* b_staged,
                          rocke_value_t* A_smem,
                          rocke_value_t* B_smem)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_coalesced_tile_loader_store_lds(b, &ctx->a_wavelet_loader, A_smem, a_staged);
    rocke_coalesced_tile_loader_store_lds(b, &ctx->b_wavelet_loader, B_smem, b_staged);
}

void rocke_conv_emit_kloop_wavelet(rocke_conv_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const int num_accs = ctx->num_accs;
    const int epi_barriers = ctx->wavelet_epi_barriers;
    int i;

    /* The wavelet driver emits the epilogue inline (inside the math branch).
     * Signal the build driver to skip its epilogue call. */
    ctx->epilogue_already_emitted = true;

    rocke_value_t* A_smem = ctx->A_smem;
    rocke_value_t* B_smem = ctx->B_smem;
    rocke_value_t* k_lo = (ctx->kloop_k_lo != NULL) ? ctx->kloop_k_lo : ctx->c0;
    rocke_value_t* k_hi = (ctx->kloop_k_hi != NULL) ? ctx->kloop_k_hi : ctx->c_K_gemm;
    rocke_value_t* c_block_k;
    rocke_value_t* c_nmath;
    rocke_value_t* warp_id_s;

    rocke_ctl_staged_t a_staged;
    rocke_ctl_staged_t b_staged;

    /* Python emits this prologue at the top of emit_wavelet_kloop_dynamic, so
     * it lands after the D descriptor. c_block_k is a fresh constant there,
     * not the one the build prologue already made. */
    c_block_k = rocke_b_const_i32(b, ctx->block_k);
    c_nmath = rocke_b_const_i32(b, ctx->wavelet_n_math_warps);
    /* warp_id is tid/wave_size -- a VGPR. Materialise it as a scalar via
     * readfirstlane so the branch lowers to s_cmp + s_cbranch (uniform) rather
     * than v_cmpx (exec-masked), which would make barrier placement inside the
     * branch accidentally legal. */
    warp_id_s = rocke_b_readfirstlane(b, ctx->warp_id);
    ctx->wavelet_is_math = rocke_b_cmp_lt(b, warp_id_s, c_nmath);
    ctx->wavelet_load_tid
        = rocke_b_sub(b, ctx->tid, rocke_b_const_i32(b, ctx->wavelet_math_block_size));

    if(ctx->is_wmma)
    {
        /* ----------------------------------------------------------------
         * WMMA path: scf_if_else (gfx1250).
         *
         * gfx1250 has separate VMEM and WMMA issue slots, so load and math
         * waves truly overlap without exec-mask interleaving. The LLVM br i1
         * divergent branch is fine here — the hardware concurrency comes
         * from the distinct issue queues, not from hardware scheduling at
         * s_barrier.
         * ---------------------------------------------------------------- */
        rocke_if_else_t ife = rocke_b_scf_if_else(b, ctx->wavelet_is_math);

        /* ---- MATH WAVE branch ---- */
        rocke_value_t* new_accs[ROCKE_CONV_MAX_ACCS];

        rocke_b_region_enter(b, ife.then_region);
        {
            /* AOT: the reduction extent is a kernarg, so the compile-time
             * peel of the final tile is gone. Every iteration now emits the
             * full barrier pair, and the load branch below runs the same trip
             * count -- guarding the barriers instead would make the two wave
             * groups emit different counts and hang the workgroup. */
            rocke_iter_arg_t m_args[ROCKE_CONV_MAX_ACCS];
            rocke_for_t for_m;
            rocke_value_t* m_vars[ROCKE_CONV_MAX_ACCS];

            rocke_b_sync(b); /* barrier_0 */

            for(i = 0; i < num_accs; ++i)
            {
                m_args[i].name = ctx->acc_names[i];
                m_args[i].init = ctx->acc_inits[i];
            }
            for_m = rocke_b_scf_for_iter(b,
                                         k_lo,
                                         k_hi,
                                         c_block_k,
                                         m_args,
                                         num_accs,
                                         "k_math",
                                         /*unroll=*/false,
                                         /*elide_trailing_barrier=*/true);
            for(i = 0; i < for_m.num_iter_vars; ++i)
                m_vars[i] = for_m.iter_vars[i];

            rocke_b_region_enter(b, for_m.body);
            {
                ctx->k_off_capture = for_m.iv;
                rocke_conv_emit_mfma_phase(
                    ctx, A_smem, B_smem, m_vars, for_m.num_iter_vars, new_accs);
                rocke_b_sync(b); /* barrier_A */
                rocke_b_sync(b); /* barrier_B */
                rocke_b_scf_yield(b, new_accs, for_m.num_iter_vars);
            }
            rocke_b_region_leave(b);

            rocke_conv_set_final_accs(ctx, for_m.op->results, for_m.op->num_results);
            rocke_conv_emit_epilogue(ctx);
        }
        rocke_b_region_leave(b);

        /* ---- LOAD WAVE branch ---- */
        rocke_b_region_enter(b, ife.else_region);
        {
            /* fetch tile k_lo -> regs, store -> LDS, barrier_0 */
            rocke_for_t for_l;

            wavelet_fetch(ctx, k_lo, &a_staged, &b_staged);
            rocke_b_s_waitcnt(b, 0, -1, -1); /* vmcnt=0 */
            wavelet_store(ctx, &a_staged, &b_staged, A_smem, B_smem);
            rocke_b_s_waitcnt(b, -1, 0, -1); /* lgkmcnt=0 */
            rocke_b_sync(b); /* barrier_0 */

            /* Same trip count as the math branch -- that is what keeps the two
             * wave groups' barrier counts equal. The trailing iteration
             * prefetches a tile past the extent; its coords fall outside the
             * descriptor bounds, so the buffer resource returns zero. */
            for_l = rocke_b_scf_for(b, k_lo, k_hi, c_block_k, "k_load");
            rocke_b_region_enter(b, for_l.body);
            {
                wavelet_fetch(ctx, rocke_b_add(b, for_l.iv, c_block_k), &a_staged, &b_staged);
                rocke_b_sync(b); /* barrier_A */
                rocke_b_s_waitcnt(b, 0, -1, -1); /* vmcnt=0 */
                wavelet_store(ctx, &a_staged, &b_staged, A_smem, B_smem);
                rocke_b_s_waitcnt(b, -1, 0, -1); /* lgkmcnt=0 */
                rocke_b_sync(b); /* barrier_B */
            }
            rocke_b_region_leave(b);

            /* epilogue stub: N_epi bare barriers matching the math branch */
            for(i = 0; i < epi_barriers; ++i)
                rocke_b_sync(b);
        }
        rocke_b_region_leave(b);
    }
    else
    {
        /* The exec-mask MFMA wavelet path has a data race: the math section reads
         * a single-buffer LDS that the load section overwrites every K iteration.
         * True concurrent execution (different SIMD units) is required, but this
         * driver emits flat sequential code with no cross-section barriers, so
         * math reads garbage after iteration 0.
         * is_valid_spec blocks wavelet on MFMA targets, so this branch is
         * unreachable in normal use. Emit an error and bail. */
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "pipeline='wavelet' is not supported for MFMA/CDNA targets: "
                        "the exec-mask split path has a data race (single LDS buffer "
                        "overwritten each K iteration with no cross-section barriers).");
    }
}
