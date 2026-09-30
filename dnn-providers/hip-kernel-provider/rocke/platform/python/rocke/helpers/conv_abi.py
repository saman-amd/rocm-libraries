# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The single source of truth for the AOT convolution kernarg ABI.

An AOT conv kernel takes its problem shape as kernel arguments, so three
places have to agree on one ordered list of names:

1. the kernel builders in ``library/kernels/common/conv_implicit_gemm*.py``,
   which emit ``b.param(...)`` in declaration order;
2. :mod:`rocke.helpers.manifest`, which turns that order into the launch
   signature;
3. ``kernels.common.conv_args`` (in the library tree), which produces the
   ``values`` dict.

Kernargs are packed *positionally* from the signature
(:func:`rocke.runtime.packing.pack_args` only uses names to look values up),
so a disagreement is not a missing-key error — it silently shifts every
argument past the first divergence and the kernel reads garbage. Declaring
the order once here and asserting it from
``library/tests/test_conv_abi.py`` is what keeps that from happening.

Ordering rule
-------------
``[pointers, byte sizes, AOT problem block, direction-specific extras]``.
The AOT block is a fixed-offset prefix shared by every variant of a
direction, so optional trailing args (``ws_ptr``, ``ks``, ``sub_gemm_buf``,
...) can be added without moving anything before them.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

__all__ = [
    "ArgSpec",
    "conv_arg_names",
    "conv_desc_names",
    "conv_direct_arg_names",
    "conv_fwd_problem_block",
]

# Pointer triple + byte-size triple that opens every direction's list.
_LEADING_ARGS = 6

# ``(name, kind)`` where kind is one of:
#   "a" / "b" / "d"  -- global pointer carrying the matching operand dtype
#   "f32*" / "i32*"  -- global pointer with a fixed element type
#   "i32"            -- scalar
ArgSpec = Tuple[str, str]


def _dims(is_3d: bool) -> List[ArgSpec]:
    """Problem extents, filter extents, and the conv attributes."""
    out: List[ArgSpec] = [
        ("p_N", "i32"),
        ("p_Hi", "i32"),
        ("p_Wi", "i32"),
        ("p_C", "i32"),
        ("p_K", "i32"),
        ("p_Y", "i32"),
        ("p_X", "i32"),
    ]
    if is_3d:
        out += [("p_Z", "i32"), ("p_Di", "i32")]
    out += [
        ("p_sH", "i32"),
        ("p_sW", "i32"),
        ("p_pH", "i32"),
        ("p_pW", "i32"),
        ("p_dH", "i32"),
        ("p_dW", "i32"),
    ]
    if is_3d:
        out += [("p_sD", "i32"), ("p_pD", "i32"), ("p_dD", "i32")]
    out += [("p_groups", "i32"), ("p_Ho", "i32"), ("p_Wo", "i32")]
    if is_3d:
        out += [("p_Do", "i32")]
    out += [("p_cpg", "i32"), ("p_kpg", "i32")]
    return out


def _magic(prefix: str, names: List[str]) -> List[ArgSpec]:
    """A ``(mult, shift)`` pair per divisor.

    One pair per divisor of the corresponding ``unmerge_magic`` step. The
    divisors are the *individual* extents, not their products:
    ``UnmergeMagicDiv`` peels one extent at a time from the last lower coord
    backwards (``tmp //= dims[i]``), so ``m -> (n, ho, wo)`` divides by
    ``Wo`` and then by ``Ho`` — never by ``Ho*Wo``.
    """
    out: List[ArgSpec] = []
    for n in names:
        out.append((f"p_magic_{prefix}{n}_mult", "i32"))
        out.append((f"p_magic_{prefix}{n}_shift", "i32"))
    return out


def _grid() -> List[ArgSpec]:
    return [("p_num_pid_m", "i32"), ("p_num_pid_n", "i32")]


def _fwd_arg_names(*, is_3d: bool = False) -> List[ArgSpec]:
    """Forward implicit-GEMM conv: ``A`` (N[D]HWC), ``B`` (K[Z]YXC), ``D`` (N[D]HWK)."""
    out: List[ArgSpec] = [
        ("A", "a"),
        ("B", "b"),
        ("D", "d"),
        ("A_bytes", "i32"),
        ("B_bytes", "i32"),
        ("D_bytes", "i32"),
    ]
    out += _dims(is_3d)
    # K_gemm = [Z*]Y*X*cpg (reduction), M = N*[Do*]Ho*Wo (output spatial).
    out += [("p_K_gemm", "i32"), ("p_M", "i32")]
    # Row-major strides in elements.
    out += [("p_A_stride_n", "i32")]
    if is_3d:
        out += [("p_A_stride_di", "i32")]
    out += [("p_A_stride_hi", "i32"), ("p_A_stride_wi", "i32")]
    out += [("p_B_stride_k", "i32")]
    if is_3d:
        out += [("p_B_stride_z", "i32")]
    out += [("p_B_stride_y", "i32"), ("p_B_stride_x", "i32")]
    out += [("p_D_stride_n", "i32")]
    if is_3d:
        out += [("p_D_stride_do", "i32")]
    out += [("p_D_stride_ho", "i32"), ("p_D_stride_wo", "i32")]
    # m -> (n, [do,] ho, wo): divisors Wo, Ho[, Do].
    out += _magic("m_", (["Do"] if is_3d else []) + ["Ho", "Wo"])
    # k -> ([z,] y, x, c): divisors cpg, X[, Y].
    out += _magic("k_", (["Y"] if is_3d else []) + ["X", "cpg"])
    out += _grid()
    return out


def conv_fwd_problem_block(*, is_3d: bool = False) -> List[ArgSpec]:
    """The forward AOT problem block on its own, without the leading pointers.

    A fused kernel that embeds the forward conv body (``deep_fused_conv_pool``)
    keeps its own pointer convention and appends this block, so it picks up the
    same runtime problem args in the same order.
    """
    return _fwd_arg_names(is_3d=is_3d)[_LEADING_ARGS:]


def _wgrad_arg_names(
    *,
    is_3d: bool = False,
    two_stage: bool = False,
) -> List[ArgSpec]:
    """Weight-gradient conv: ``dY`` (N[D]HWK), ``X`` (N[D]HWC), ``dW`` (K[Z]YXC).

    GEMM mapping: ``M = kpg``, ``N = [Z*]Y*X*cpg``, ``K = N*[Do*]Ho*Wo``.

    ``ks`` (slice width) and ``ks_count`` (number of slices) are always
    present. The split-K degree is a launch parameter, never a compile-time
    constant, so one binary serves every degree and the block-z decode is the
    same shape whether the kernel is split or not:

        group = z // ks_count ;  slice = z % ks_count
        k_lo  = slice * ks    ;  k_hi  = k_lo + ks

    An unsplit launch passes ``ks_count = 1`` and ``ks = wg_K``, which
    collapses that to ``group = z``, ``k_lo = 0``, ``k_hi = wg_K``. Making the
    two cases one removes the pair of ABI flags that used to select between
    them -- and with them the chance of building a kernel against one variant
    and launching it with the other.
    """
    out: List[ArgSpec] = [
        ("dY", "a"),
        ("X", "b"),
        ("dW", "d"),
        ("dY_bytes", "i32"),
        ("X_bytes", "i32"),
        ("dW_bytes", "i32"),
    ]
    out += _dims(is_3d)
    out += [("p_wg_M", "i32"), ("p_wg_N", "i32"), ("p_wg_K", "i32")]
    out += [("p_dY_stride_n", "i32")]
    if is_3d:
        out += [("p_dY_stride_do", "i32")]
    out += [("p_dY_stride_ho", "i32"), ("p_dY_stride_wo", "i32")]
    out += [("p_X_stride_n", "i32")]
    if is_3d:
        out += [("p_X_stride_di", "i32")]
    out += [("p_X_stride_hi", "i32"), ("p_X_stride_wi", "i32")]
    out += [("p_dW_stride_k", "i32")]
    if is_3d:
        out += [("p_dW_stride_z", "i32")]
    out += [("p_dW_stride_y", "i32"), ("p_dW_stride_x", "i32")]
    # k_wg -> (n, [do,] ho, wo): divisors Wo, Ho[, Do].
    out += _magic("k_", (["Do"] if is_3d else []) + ["Ho", "Wo"])
    # n_wg -> ([z,] y, x, c): divisors cpg, X[, Y].
    out += _magic("n_", (["Y"] if is_3d else []) + ["X", "cpg"])
    out += _grid()
    # ---- variant-specific extras ----
    if two_stage:
        out += [("ws_ptr", "f32*"), ("ws_bytes", "i32")]
    # The split-K slice width and degree, always. `ks` alone is not enough:
    # decoding block_id_z needs the number of slices, which is a different
    # quantity from how wide each one is.
    out += [("ks", "i32"), ("ks_count", "i32")]
    return out


def _dgrad_arg_names(*, is_3d: bool = False) -> List[ArgSpec]:
    """Data-gradient conv: ``dY`` (N[D]HWK), ``W`` (K[Z]YXC), ``dX`` (N[D]HWC).

    GEMM mapping: ``M = N*[Di*]Hi*Wi``, ``N = cpg``, ``K = [Z*]Y*X*kpg``.

    The 3-D form is declared here even though ``DgradConvSpec.validate()``
    still rejects 3-D: the ABI is the contract, and writing it down once --
    laid out the same way the forward and wgrad blocks are -- is what lets the
    kernel be built against a fixed target rather than having the argument
    order invented alongside it. Until the kernel lands, asking for
    ``is_3d=True`` yields a signature nothing can be launched with, which is
    the honest answer; the spec validator is what refuses the build.
    """
    out: List[ArgSpec] = [
        ("dY", "a"),
        ("W", "b"),
        ("dX", "d"),
        ("dY_bytes", "i32"),
        ("W_bytes", "i32"),
        ("dX_bytes", "i32"),
    ]
    out += _dims(is_3d)
    out += [("p_dg_M", "i32"), ("p_dg_N", "i32"), ("p_dg_K", "i32")]
    out += [("p_dY_stride_n", "i32")]
    if is_3d:
        out += [("p_dY_stride_do", "i32")]
    out += [("p_dY_stride_ho", "i32"), ("p_dY_stride_wo", "i32")]
    out += [("p_W_stride_k", "i32")]
    if is_3d:
        out += [("p_W_stride_z", "i32")]
    out += [("p_W_stride_y", "i32"), ("p_W_stride_x", "i32")]
    out += [("p_dX_stride_n", "i32")]
    if is_3d:
        out += [("p_dX_stride_di", "i32")]
    out += [("p_dX_stride_hi", "i32"), ("p_dX_stride_wi", "i32")]
    # Only the ``m`` decode needs magic pairs, for the dX store descriptor:
    # ``m -> (n, [di,] hi, wi)`` divides by Wi, then Hi[, then Di]. The dY / W
    # load descriptors decode ``m`` and ``k`` with plain runtime div/mod
    # against fields of the tilde sub-GEMM record, not through an
    # unmerge_magic chain, so they have no divisor to precompute a pair for.
    out += _magic("m_", (["Di"] if is_3d else []) + ["Hi", "Wi"])
    out += _grid()
    # The tilde decomposition record buffer is always present.
    out += [("sub_gemm_buf", "i32*"), ("num_sub_gemms", "i32")]
    return out


def conv_arg_names(
    *,
    direction: str = "fwd",
    is_3d: bool = False,
    two_stage: bool = False,
) -> List[ArgSpec]:
    """The implicit-GEMM conv kernarg ABI for one direction.

    ``direction`` is ``"fwd"``, ``"wgrad"`` or ``"dgrad"``. All three share
    the pointer triple, the byte triple and the extent block, and differ only
    in which GEMM dims, strides and magic families follow -- so taking the
    direction as data rather than as three separate entry points lets the
    callers that already dispatch on it (the manifest emitter, the grouped
    dispatcher, the AOT cache) stop branching.

    ``two_stage`` belongs to wgrad's deterministic epilogue and is rejected
    elsewhere rather than ignored: silently accepting it would hand back a
    signature that does not match any kernel the flag can build.
    """
    if two_stage and direction != "wgrad":
        raise ValueError(
            f"two_stage is a wgrad-only variant (got direction={direction!r})"
        )
    if direction == "fwd":
        return _fwd_arg_names(is_3d=is_3d)
    if direction == "wgrad":
        return _wgrad_arg_names(is_3d=is_3d, two_stage=two_stage)
    if direction == "dgrad":
        return _dgrad_arg_names(is_3d=is_3d)
    raise ValueError(
        f"unknown direction {direction!r}; expected 'fwd', 'wgrad' or 'dgrad'"
    )


def conv_direct_arg_names(*, direction: str = "fwd") -> List[ArgSpec]:
    """Direct grouped convolution: ``A`` (NHWC), ``B`` (KRSC), ``D`` (NHWK).

    Direct conv splits its parameters differently from implicit GEMM. The
    filter extents (``KH``, ``KW``), ``stride``, ``PAD`` and the per-group
    channel counts (``cpg``, ``kpg``) stay build-time: they shape the unrolled
    MFMA chain, the LDS row layout and the tap offsets, so they are kernel
    *capabilities*, the same kind of thing as a tile size. The AOT cache keys
    on them and refuses a problem that disagrees.

    Everything that merely sizes the work -- batch, spatial extents and the
    group count -- is a runtime kernarg, which is what lets one compiled
    kernel serve any image size.

    The weight tensor needs no stride args: its strides are products of
    ``KH``/``KW``/``cpg`` alone, all build-time.
    """
    # Direct conv has a forward and a data-gradient kernel and nothing else.
    # Without this check any other string fell through to the forward ABI and
    # handed back a signature for a kernel that does not exist.
    if direction not in ("fwd", "dgrad"):
        raise ValueError(
            f"direct conv has no {direction!r} kernel; expected 'fwd' or 'dgrad'"
        )
    out: List[ArgSpec] = [
        ("A", "a"),
        ("B", "b"),
        ("D", "d"),
        ("A_bytes", "i32"),
        ("B_bytes", "i32"),
        ("D_bytes", "i32"),
        ("p_N", "i32"),
        ("p_Hi", "i32"),
        ("p_Wi", "i32"),
        ("p_Ho", "i32"),
        ("p_Wo", "i32"),
        ("p_groups", "i32"),
        ("p_total_c", "i32"),  # groups * cpg
        ("p_total_k", "i32"),  # groups * kpg
    ]
    # Strides of the tensors bound to A and D. Forward runs
    # NHWC -> NHWK, so A is the activation and D the output; dgrad runs the
    # other way, so the two stride triples swap layouts. The names follow the
    # tensor, not the operand slot, so a launch-site typo is visible.
    if direction == "dgrad":
        out += [
            ("p_dY_stride_n", "i32"),  # Ho*Wo*total_k
            ("p_dY_stride_ho", "i32"),  # Wo*total_k
            ("p_dY_stride_wo", "i32"),  # total_k
            ("p_dX_stride_n", "i32"),  # Hi*Wi*total_c
            ("p_dX_stride_hi", "i32"),  # Wi*total_c
            ("p_dX_stride_wi", "i32"),  # total_c
        ]
    else:
        out += [
            ("p_A_stride_n", "i32"),  # Hi*Wi*total_c
            ("p_A_stride_hi", "i32"),  # Wi*total_c
            ("p_A_stride_wi", "i32"),  # total_c
            ("p_D_stride_n", "i32"),  # Ho*Wo*total_k
            ("p_D_stride_ho", "i32"),  # Wo*total_k
            ("p_D_stride_wo", "i32"),  # total_k
        ]
    return out


def conv_desc_names(arg_names: Sequence[ArgSpec]) -> List[str]:
    """Just the names, in order — handy for assertions and error messages."""
    return [name for name, _ in arg_names]
