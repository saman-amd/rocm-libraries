# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Guard: the AOT conv kernarg ABI agrees across all three places that use it.

Kernargs are packed *positionally* from the launch signature
(``rocke.runtime.packing.pack_args`` only uses names to look values up), so if
the kernel declares its params in a different order than the signature lists
them, every argument past the first divergence silently shifts — pointers get
read out of i32 slots and the kernel faults or corrupts memory. Nothing else
in the test suite catches that, because both sides individually look fine.

This pins together:

1. ``rocke.helpers.conv_abi``     — the ordered name list (the ABI)
2. the kernel builders                — ``[p.name for p in kernel.params]``
3. ``rocke.helpers.manifest``         — the launch signature
4. ``kernels.common.conv_args``       — the host-side ``values`` dict keys
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

import pytest

from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_implicit_gemm import (
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
)
from kernels.common.conv_implicit_gemm_dgrad import (
    DgradConvSpec,
    build_implicit_gemm_conv_dgrad,
)
from kernels.common.conv_implicit_gemm_wgrad import (
    WgradConvSpec,
    build_implicit_gemm_conv_wgrad,
)
from kernels.common.conv_args import ConvArgs
from rocke.helpers.conv_abi import conv_arg_names, conv_direct_arg_names
from rocke.helpers.manifest import conv_args_signature, conv_direct_args_signature

_ARCH = "gfx950"
_DATA = ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16")

_TILE = dict(
    tile_m=64,
    tile_n=64,
    tile_k=32,
    warp_m=2,
    warp_n=2,
    warp_tile_m=16,
    warp_tile_n=16,
    warp_tile_k=16,
    wave_size=64,
    vector_size_c=1,
)

_P2D = ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1)
_P2D_GROUPED = ConvProblem(
    N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, groups=2
)
_P3D = ConvProblem(
    N=2,
    Di=8,
    Hi=14,
    Wi=14,
    C=64,
    K=64,
    Z=3,
    Y=3,
    X=3,
    sH=1,
    sW=1,
    pH=1,
    pW=1,
    sD=1,
    pD=1,
    dD=1,
)


def _names(sig):
    return [a["name"] for a in sig]


def _assert_abi(kernel, abi_names, signature, label):
    """The kernel, the ABI list and the signature must be the same sequence."""
    kernel_order = [p.name for p in kernel.params]
    expected = [n for n, _ in abi_names]
    assert kernel_order == expected, (
        f"{label}: kernel param order diverges from conv_abi.\n"
        f"  kernel: {kernel_order}\n  abi:    {expected}"
    )
    assert _names(signature) == expected, (
        f"{label}: launch signature diverges from conv_abi.\n"
        f"  signature: {_names(signature)}\n  abi:       {expected}"
    )


# ---------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,problem,is_3d",
    [
        ("fwd-2d", _P2D, False),
        ("fwd-2d-grouped", _P2D_GROUPED, False),
        ("fwd-3d", _P3D, True),
    ],
)
def test_fwd_abi(label, problem, is_3d):
    kernel = build_implicit_gemm_conv(
        ImplicitGemmConvSpec(problem=problem, data=_DATA, **_TILE), arch=_ARCH
    )
    _assert_abi(
        kernel,
        conv_arg_names(direction="fwd", is_3d=is_3d),
        conv_args_signature("fp16", is_3d=is_3d),
        label,
    )


@pytest.mark.parametrize(
    "label,pipeline,extra",
    [
        ("mem", "mem", {}),
        ("compv3", "compv3", {}),
        ("compv4", "compv4", {}),
        ("basic", "basic", {}),
        ("unroll_k", "mem", {"unroll_k": True}),
        ("async_dma", "mem", {"async_dma": True}),
        ("chiplet", "mem", {"chiplet_swizzle": True}),
    ],
)
def test_fwd_abi_is_pipeline_independent(label, pipeline, extra):
    """The kernarg block must not depend on the K-loop driver."""
    kernel = build_implicit_gemm_conv(
        ImplicitGemmConvSpec(
            problem=_P2D, data=_DATA, pipeline=pipeline, **_TILE, **extra
        ),
        arch=_ARCH,
    )
    _assert_abi(
        kernel,
        conv_arg_names(direction="fwd"),
        conv_args_signature("fp16"),
        f"fwd-{label}",
    )


# ---------------------------------------------------------------------
# wgrad — one case per optional trailing-arg combination
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,problem,spec_kw,abi_kw",
    [
        ("wgrad-2d", _P2D, {"split_k": 1}, {}),
        ("wgrad-3d", _P3D, {"split_k": 1}, {"is_3d": True}),
        (
            "wgrad-split-k",
            _P2D,
            {"split_k": 0, "epilogue": "cshuffle", "vector_size_c": 2},
            {},
        ),
        (
            "wgrad-split-k-grouped",
            _P2D_GROUPED,
            {"split_k": 0, "epilogue": "cshuffle", "vector_size_c": 2},
            {},
        ),
    ],
)
def test_wgrad_abi(label, problem, spec_kw, abi_kw):
    tile = dict(_TILE)
    tile.update({k: v for k, v in spec_kw.items() if k in tile})
    spec_kw = {k: v for k, v in spec_kw.items() if k not in tile}
    kernel = build_implicit_gemm_conv_wgrad(
        WgradConvSpec(problem=problem, data=_DATA, **tile, **spec_kw), arch=_ARCH
    )
    _assert_abi(
        kernel,
        conv_arg_names(direction="wgrad", **abi_kw),
        conv_args_signature("fp16", direction="wgrad", **abi_kw),
        label,
    )


# ---------------------------------------------------------------------
# dgrad
# ---------------------------------------------------------------------


def test_dgrad_abi():
    kernel = build_implicit_gemm_conv_dgrad(
        DgradConvSpec(problem=_P2D, data=_DATA, **_TILE), arch=_ARCH
    )
    _assert_abi(
        kernel,
        conv_arg_names(direction="dgrad"),
        conv_args_signature("fp16", direction="dgrad"),
        "dgrad",
    )


# ---------------------------------------------------------------------
# host-side values dicts
# ---------------------------------------------------------------------


@pytest.mark.parametrize("problem", [_P2D, _P2D_GROUPED, _P3D])
def test_fwd_launch_values_cover_the_abi(problem):
    values = ConvArgs.from_problem(problem, tile_m=64, tile_n=64).to_launch_values(
        1, 2, 3, 4, 5, 6
    )
    expected = [n for n, _ in conv_arg_names(direction="fwd", is_3d=problem.is_3d)]
    assert sorted(values) == sorted(expected)


@pytest.mark.parametrize("problem", [_P2D, _P3D])
def test_wgrad_launch_values_cover_the_abi(problem):
    values = ConvArgs.from_problem(
        problem, direction="wgrad", tile_m=64, tile_n=64, tile_k=32
    ).to_launch_values(1, 2, 3, 4, 5, 6)
    expected = [n for n, _ in conv_arg_names(direction="wgrad", is_3d=problem.is_3d)]
    assert sorted(values) == sorted(expected)


def test_wgrad_launch_values_split_k_grouped():
    """The split degree is a launch parameter, so it changes values not shape.

    Splitting must not select a different ABI: the same kernel serves any
    degree, and ``ks``/``ks_count`` simply carry different numbers.
    """
    args = ConvArgs.from_problem(
        _P2D_GROUPED, direction="wgrad", tile_m=64, tile_n=64, tile_k=32
    )
    expected = sorted(n for n, _ in conv_arg_names(direction="wgrad"))
    unsplit = args.to_launch_values(1, 2, 3, 4, 5, 6, split_k=1)
    split = args.to_launch_values(1, 2, 3, 4, 5, 6, split_k=4)
    assert sorted(unsplit) == expected
    assert sorted(split) == expected
    assert unsplit["ks_count"] == 1 and split["ks_count"] == 4
    # Every slice is tile-aligned and together they cover the whole reduction.
    assert split["ks"] % 32 == 0
    assert split["ks"] * 4 >= args.gemm_k


def test_wgrad_launch_values_two_stage():
    values = ConvArgs.from_problem(
        _P2D, direction="wgrad", tile_m=64, tile_n=64, tile_k=32
    ).to_launch_values(1, 2, 3, 4, 5, 6, ws_ptr=99, ws_bytes=128)
    expected = [n for n, _ in conv_arg_names(direction="wgrad", two_stage=True)]
    assert sorted(values) == sorted(expected)


def test_dgrad_launch_values_cover_the_abi():
    values = ConvArgs.from_problem(
        _P2D, direction="dgrad", tile_m=64, tile_n=64
    ).to_launch_values(1, 2, 3, 4, 5, 6, sub_gemm_buf=7, num_sub_gemms=1)
    expected = [n for n, _ in conv_arg_names(direction="dgrad")]
    assert sorted(values) == sorted(expected)


def test_dgrad_3d_abi_exists_but_the_spec_refuses_to_build_it():
    """The 3-D dgrad ABI is declared; only the spec validator says no.

    Keeping the refusal in exactly one place matters: if the argument builder
    also refused, a future 3-D dgrad kernel would land and silently have no
    way to be launched, and the second refusal would be found by debugging
    rather than by reading the validator.
    """
    from kernels.common.conv_implicit_gemm_dgrad import DgradConvSpec

    # The host can fill the 3-D block, and it matches the declared ABI.
    values = ConvArgs.from_problem(
        _P3D, direction="dgrad", tile_m=64, tile_n=64
    ).to_launch_values(1, 2, 3, 4, 5, 6, sub_gemm_buf=7, num_sub_gemms=1)
    assert sorted(values) == sorted(
        n for n, _ in conv_arg_names(direction="dgrad", is_3d=True)
    )

    # Building a kernel for it is what fails, and it says why.
    with pytest.raises(ValueError, match="3-D"):
        DgradConvSpec(problem=_P3D, data=_DATA, **_TILE).validate()


# ---------------------------------------------------------------------
# magic-number semantics
# ---------------------------------------------------------------------


def _magic_div(x: int, mult: int, shift: int) -> int:
    """Mirror of ``do_magic_division`` on unsigned 32-bit values."""
    if mult < 0:
        mult += 1 << 32
    return (((x * mult) >> 32) + x) >> shift


@pytest.mark.parametrize("problem", [_P2D, _P2D_GROUPED])
def test_m_decode_matches_the_reference_split(problem):
    """``m -> (n, ho, wo)`` must divide by Wo then Ho, not by Ho*Wo.

    Dividing by the product instead collapses every batch index onto n == 0,
    which is correct only for N == 1 — exactly the shape a smoke test uses.
    """
    args = ConvArgs.from_problem(problem, tile_m=64, tile_n=64)
    v = args.to_launch_values(0, 0, 0, 0, 0, 0)
    Ho, Wo, N = v["p_Ho"], v["p_Wo"], v["p_N"]
    for m in range(N * Ho * Wo):
        tmp = m
        wo = tmp - _magic_div(tmp, v["p_magic_m_Wo_mult"], v["p_magic_m_Wo_shift"]) * Wo
        tmp = _magic_div(tmp, v["p_magic_m_Wo_mult"], v["p_magic_m_Wo_shift"])
        ho = tmp - _magic_div(tmp, v["p_magic_m_Ho_mult"], v["p_magic_m_Ho_shift"]) * Ho
        n = _magic_div(tmp, v["p_magic_m_Ho_mult"], v["p_magic_m_Ho_shift"])
        assert (n, ho, wo) == (m // (Ho * Wo), (m // Wo) % Ho, m % Wo), f"m={m}"


def test_k_decode_matches_the_reference_split():
    """``k -> (y, x, c)`` must divide by cpg then X, not by X*cpg."""
    args = ConvArgs.from_problem(_P2D, tile_m=64, tile_n=64)
    v = args.to_launch_values(0, 0, 0, 0, 0, 0)
    X, cpg, Y = v["p_X"], v["p_cpg"], v["p_Y"]
    for k in range(Y * X * cpg):
        q = _magic_div(k, v["p_magic_k_cpg_mult"], v["p_magic_k_cpg_shift"])
        c = k - q * cpg
        y = _magic_div(q, v["p_magic_k_X_mult"], v["p_magic_k_X_shift"])
        x = q - y * X
        assert (y, x, c) == (k // (X * cpg), (k // cpg) % X, k % cpg), f"k={k}"


# ---------------------------------------------------------------------
# direct grouped convolution
# ---------------------------------------------------------------------


def _op_signature(kernel):
    """Structural fingerprint of the emitted IR: param names + op sequence.

    Two kernels with the same fingerprint emit the same instructions, which is
    the property that makes a kernel shape-generic.

    Constant *values* are part of the fingerprint, not just the op names. A
    baked ``const_i32(Ho)`` emits the same ``arith.constant`` op whatever the
    shape, so comparing op names alone would call such a kernel shape-generic
    while it silently only computes the right answer for the height it was
    built for -- exactly the bug this test exists to catch.
    """
    out = [p.name for p in kernel.params]

    def walk(op):
        if op.name == "arith.constant":
            out.append(f"{op.name}={op.attrs.get('value')!r}")
        else:
            out.append(op.name)
        for region in op.regions or []:
            for inner in region.ops:
                walk(inner)

    for op in kernel.body.ops:
        walk(op)
    return tuple(out)


def _direct_builders():
    """``(label, channel kwargs, direction, build fn)`` for every direct variant."""
    from kernels.common import conv_direct_grouped as dc

    return [
        (
            "16c",
            dict(cpg=16, kpg=16, groups=8),
            "fwd",
            lambda pr: dc.build_direct_conv_16c(
                dc.DirectConv16cSpec(problem=pr), arch=_ARCH
            ),
        ),
        (
            "4c",
            dict(cpg=4, kpg=4, groups=32),
            "fwd",
            lambda pr: dc.build_direct_conv_4c(
                dc.DirectConv4cSpec(problem=pr), arch=_ARCH
            ),
        ),
        (
            "8c",
            dict(cpg=8, kpg=8, groups=8),
            "fwd",
            lambda pr: dc.build_direct_conv_8c(
                dc.DirectConv8cSpec(problem=pr), arch=_ARCH
            ),
        ),
        (
            "32c",
            dict(cpg=32, kpg=32, groups=4),
            "fwd",
            lambda pr: dc.build_direct_conv_32c(
                dc.DirectConv32cSpec(problem=pr), arch=_ARCH
            ),
        ),
        (
            "generic",
            dict(cpg=16, kpg=16, groups=8),
            "fwd",
            lambda pr: dc.build_direct_conv(dc.DirectConvSpec(problem=pr), arch=_ARCH),
        ),
        (
            "depthwise",
            dict(cpg=1, kpg=1, groups=32),
            "fwd",
            lambda pr: dc.build_direct_depthwise(
                dc.DirectDepthwiseSpec(problem=pr), arch=_ARCH
            ),
        ),
        (
            "depthwise_spatial",
            dict(cpg=1, kpg=1, groups=32),
            "fwd",
            lambda pr: dc.build_direct_depthwise_spatial(
                dc.DirectDepthwiseSpatialSpec(problem=pr), arch=_ARCH
            ),
        ),
        (
            "dgrad",
            dict(cpg=16, kpg=16, groups=8),
            "dgrad",
            lambda pr: dc.build_direct_conv_dgrad(
                dc.DirectConvDgradSpec(problem=pr), arch=_ARCH
            ),
        ),
        (
            "depthwise_dgrad",
            dict(cpg=1, kpg=1, groups=32),
            "dgrad",
            lambda pr: dc.build_direct_depthwise_dgrad(
                dc.DirectDepthwiseDgradSpec(problem=pr), arch=_ARCH
            ),
        ),
        (
            "depthwise_dgrad_streaming",
            dict(cpg=1, kpg=1, groups=32),
            "dgrad",
            lambda pr: dc.build_direct_depthwise_dgrad_streaming(
                dc.DirectDepthwiseDgradStreamSpec(problem=pr), arch=_ARCH
            ),
        ),
        # The MFMA dgrad's compute pass: a forward kernel on the transposed
        # problem, so it takes the forward ABI.
        (
            "mfma_dgrad_fprop",
            dict(cpg=16, kpg=64, groups=8),
            "fwd",
            lambda pr: dc.build_direct_conv(
                dc.make_dgrad_fprop_spec(pr, block_h=8, waves_k=4), arch=_ARCH
            ),
        ),
    ]


# The spatial depthwise kernel packs every channel of an image into one
# wavefront, so its lane layout -- and with it the emitted IR -- follows the
# group count. It is the one direct kernel whose groups are not a kernarg.
_GROUPS_BAKED = {"depthwise_spatial"}


@pytest.mark.parametrize(
    "label,channels,direction,build",
    _direct_builders(),
    ids=[c[0] for c in _direct_builders()],
)
def test_direct_conv_abi_and_shape_invariance(label, channels, direction, build):
    """Direct conv must be shape-generic and match the direct kernarg ABI.

    The shape-invariance half is the load-bearing one: these builders used to
    unroll the H-row loop at build time, so a kernel silently only worked for
    the height it was compiled for. Comparing the emitted op sequence across
    three unrelated shapes is what catches a regression back to that.
    """
    from kernels.common.conv_direct_grouped import DirectConvProblem
    from rocke.helpers.conv_abi import conv_direct_arg_names
    from rocke.helpers.manifest import conv_direct_args_signature

    fingerprints = []
    kernel = None
    # The group count varies too: it is a kernarg, so a binary built for one
    # count has to serve every multiple of the block's group tile.
    for (N, H, W), group_mult in zip(
        [(2, 14, 14), (3, 29, 37), (1, 64, 64)], (1, 2, 3)
    ):
        shape = dict(channels)
        if label not in _GROUPS_BAKED:
            shape["groups"] *= group_mult
        problem = DirectConvProblem(N=N, H=H, W=W, KH=3, KW=3, PAD=1, stride=1, **shape)
        kernel = build(problem)
        fingerprints.append(_op_signature(kernel))

    assert len(set(fingerprints)) == 1, (
        f"{label}: emitted IR depends on the problem shape, so the kernel is "
        f"not shape-generic"
    )

    expected = [n for n, _ in conv_direct_arg_names(direction=direction)]
    assert [p.name for p in kernel.params] == expected, f"{label}: kernarg order"
    signature = conv_direct_args_signature("fp16", direction=direction)
    assert _names(signature) == expected, f"{label}: launch signature"


@pytest.mark.parametrize("fold_k32", [False, True])
def test_direct_dgrad_weight_transforms_are_shape_invariant(fold_k32):
    """The MFMA dgrad's weight transforms are cached once per filter/channels.

    They bake KH/KW/cpg/kpg and nothing else, so the same binary has to come
    out for any batch, image size or group count.
    """
    from kernels.common import conv_direct_grouped as dc

    fingerprints = set()
    for N, H, W, groups in [(2, 14, 14, 8), (3, 29, 37, 16), (1, 64, 64, 24)]:
        problem = dc.DirectConvProblem(
            N=N, H=H, W=W, groups=groups, cpg=32, kpg=32, KH=3, KW=3, PAD=1, stride=1
        )
        transpose = dc.build_direct_transpose_weights_dgrad(
            dc.DirectTransposeWeightsDgradSpec(problem=problem), arch=_ARCH
        )
        reorganize = dc.build_direct_reorganize_weights(
            dc.DirectReorganizeWeightsSpec(problem=problem, fold_k32=fold_k32),
            arch=_ARCH,
        )
        fingerprints.add((_op_signature(transpose), _op_signature(reorganize)))
    assert len(fingerprints) == 1


@pytest.mark.parametrize("direction", ["fwd", "dgrad"])
def test_direct_launch_values_cover_the_abi(direction):
    from kernels.common.conv_direct_grouped import DirectConvProblem
    from kernels.common.conv_args import ConvArgs
    from rocke.helpers.conv_abi import conv_direct_arg_names

    problem = DirectConvProblem(N=2, H=14, W=14, groups=8, cpg=16, kpg=16)
    values = ConvArgs.from_problem(problem, direction=direction).to_launch_values(
        1, 2, 3, 4, 5, 6
    )
    expected = [n for n, _ in conv_direct_arg_names(direction=direction)]
    assert sorted(values) == sorted(expected)


# ---------------------------------------------------------------------
# implicit GEMM: shape invariance
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,build",
    [
        (
            "fwd",
            lambda pr: build_implicit_gemm_conv(
                ImplicitGemmConvSpec(problem=pr, data=_DATA, **_TILE), arch=_ARCH
            ),
        ),
        (
            "wgrad",
            lambda pr: build_implicit_gemm_conv_wgrad(
                WgradConvSpec(problem=pr, data=_DATA, split_k=1, **_TILE), arch=_ARCH
            ),
        ),
        (
            "dgrad",
            lambda pr: build_implicit_gemm_conv_dgrad(
                DgradConvSpec(problem=pr, data=_DATA, **_TILE), arch=_ARCH
            ),
        ),
    ],
)
def test_implicit_gemm_is_shape_invariant(label, build):
    """One compiled kernel per tile config must serve every compatible shape."""
    shapes = [
        ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1),
        ConvProblem(N=5, Hi=31, Wi=27, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1),
        ConvProblem(N=1, Hi=64, Wi=64, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1),
    ]
    fingerprints = {_op_signature(build(p)) for p in shapes}
    assert len(fingerprints) == 1, f"{label}: emitted IR depends on the shape"


# ---------------------------------------------------------------------
# capability coverage: what the IR *does* depend on must be in the identity
# ---------------------------------------------------------------------


_CAPABILITY_CASES = [
    # (label, build fn, attribute kwargs, KernelIdentity fields that record it)
    ("fwd grouped", "fwd", dict(groups=2), ("grouped",)),
    ("wgrad grouped", "wgrad", dict(groups=2), ("grouped",)),
    ("dgrad grouped", "dgrad", dict(groups=2), ("grouped",)),
    ("dgrad stride", "dgrad", dict(sH=2, sW=2), ("stride_h", "stride_w")),
    ("dgrad dilation", "dgrad", dict(dH=2, dW=2), ("dilation_h", "dilation_w")),
]


def _build_direction(direction, problem):
    if direction == "fwd":
        return build_implicit_gemm_conv(
            ImplicitGemmConvSpec(problem=problem, data=_DATA, **_TILE), arch=_ARCH
        )
    if direction == "wgrad":
        return build_implicit_gemm_conv_wgrad(
            WgradConvSpec(problem=problem, data=_DATA, split_k=1, **_TILE), arch=_ARCH
        )
    return build_implicit_gemm_conv_dgrad(
        DgradConvSpec(problem=problem, data=_DATA, **_TILE), arch=_ARCH
    )


@pytest.mark.parametrize(
    "label,direction,attrs,identity_fields",
    _CAPABILITY_CASES,
    ids=[c[0] for c in _CAPABILITY_CASES],
)
def test_baked_attributes_are_recorded_as_capabilities(
    label, direction, attrs, identity_fields
):
    """Anything the IR bakes must be a capability the AOT cache can filter on.

    A problem attribute that changes the emitted IR but is *not* in
    :class:`~benchmarks.common.kernel_cache.KernelIdentity` is the worst kind
    of AOT bug: the
    cache hands out a kernel built for a different value, the launch succeeds,
    and the numbers are quietly wrong. Two halves are asserted here --
    the attribute really does change the IR (so it is not over-declared), and
    the identity has a field for it (so the cache can reject the mismatch).
    """
    from dataclasses import fields as dc_fields

    from benchmarks.common.kernel_cache import KernelIdentity

    base = ConvProblem(N=2, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1)
    variant = dc_replace(base, **attrs)

    assert _op_signature(_build_direction(direction, base)) != _op_signature(
        _build_direction(direction, variant)
    ), (
        f"{label}: this attribute no longer changes the emitted IR. If that is "
        f"intentional, drop it from the identity so the cache stops splitting "
        f"binaries on it."
    )

    known = {f.name for f in dc_fields(KernelIdentity)}
    missing = [f for f in identity_fields if f not in known]
    assert not missing, f"{label}: KernelIdentity has no field(s) {missing}"


def test_cache_rejects_capability_mismatches():
    """``supports_problem`` must refuse a kernel built for other capabilities.

    The identity carrying the field is only half the guard; the cache has to
    actually check it. Each case below produced a silently wrong result before
    the check existed.
    """
    from benchmarks.common.kernel_cache import KernelCache, KernelIdentity

    cache = KernelCache("/nonexistent-cache-dir-for-a-pure-predicate-test", _ARCH)
    common = dict(
        arch=_ARCH,
        algorithm="implicit_gemm",
        dtype_a="fp16",
        dtype_b="fp16",
        dtype_d="fp16",
        tile_m=32,
        tile_n=32,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        pipeline="mem",
        epilogue="cshuffle",
        wave_size=64,
        vector_size_a=2,
        vector_size_b=2,
        vector_size_c=2,
    )
    plain = ConvProblem(N=1, Hi=16, Wi=16, C=32, K=32, Y=3, X=3, pH=1, pW=1)
    strided = dc_replace(plain, sH=2, sW=2)
    dilated = dc_replace(plain, dH=2, dW=2)
    grouped = dc_replace(plain, groups=2)

    ungrouped_fwd = KernelIdentity(direction="fwd", grouped=False, **common)
    grouped_fwd = KernelIdentity(direction="fwd", grouped=True, **common)
    dgrad_s1 = KernelIdentity(
        direction="dgrad",
        stride_h=1,
        stride_w=1,
        dilation_h=1,
        dilation_w=1,
        max_sub_gemms=64,
        **common,
    )

    assert cache.supports_problem(ungrouped_fwd, plain)[0]
    assert cache.supports_problem(grouped_fwd, grouped)[0]
    assert not cache.supports_problem(ungrouped_fwd, grouped)[0]
    assert not cache.supports_problem(grouped_fwd, plain)[0]

    assert cache.supports_problem(dgrad_s1, plain)[0]
    assert not cache.supports_problem(dgrad_s1, strided)[0]
    assert not cache.supports_problem(dgrad_s1, dilated)[0]
