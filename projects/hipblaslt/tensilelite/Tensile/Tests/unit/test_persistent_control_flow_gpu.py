# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execute the extracted persistent control-flow emitters in small GPU kernels."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from rocisa.code import Module
from rocisa.container import ContinuousRegister, sgpr
from rocisa.instruction import SCmpEQU32

from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.Components.TileProcessingStrategy import TileProcessingStrategy
from Tensile.Components.WorkAssignment import _extract_hybrid_mode
from gpu_test_helpers import GPU_MARKS, init_rocisa, run_scalar_kernel

pytestmark = [pytest.mark.unit, *GPU_MARKS]


@pytest.fixture(autouse=True)
def _gpu_instruction_set(_isolate_rocisa_state):
    init_rocisa(wavesize=64)


@pytest.mark.parametrize("k,tail,max_unit,gsu,runtime_gsu,expected", [
    (0, False, 1, 0, 1, 0),
    (16, False, 1, 0, 1, 1),
    (31, False, 1, 0, 1, 1),
    (0, True, 1, 0, 1, 0),
    (16, True, 1, 0, 1, 1),
    (17, True, 1, 0, 1, 2),
    (20, True, 4, 0, 1, 2),
    (18, True, 4, 0, 1, 1),
    (17, True, 1, 2, 2, 1),
    (17, True, 1, 2, 1, 2),
])
def test_ordinary_loop_count_respects_tail_and_gsu(
    tmp_path, k, tail, max_unit, gsu, runtime_gsu, expected,
):
    writer = SimpleNamespace(
        states=SimpleNamespace(tailloopInNll=tail, tailloopInNllmaxUnit=max_unit, unrollIdx=0),
        gsuMaskHex=lambda kernel: "0x3fff",
        sgprs={"LoopCounter": 6, "SizesSum": 8, "GSU": 9},
    )
    module = KernelWriterAssembly._calculateOrdinaryLoopNumIter(
        writer, {"DepthU": 16, "GlobalSplitU": gsu}, "LoopCounter", 0,
        ContinuousRegister(32, 3),
    )
    observed = run_scalar_kernel(
        writer, module, {"SizesSum": k, "GSU": runtime_gsu}, ["LoopCounter"], tmp_path,
    )
    assert observed["LoopCounter"] == (expected,) * 64


@pytest.mark.parametrize("gsu,runtime_gsu,arg_type,keep_address", [
    (0, 1, 0, True),
    (0, 1, 3, False),
    (2, 1, 3, False),
    (2, 2, 3, True),
])
def test_ordinary_srd_routes_general_batch_and_gsu(
    monkeypatch, tmp_path, gsu, runtime_gsu, arg_type, keep_address,
):
    def reject_strategy_lookup(*args, **kwargs):
        pytest.fail("Ordinary writer emission must not request a persistent strategy")

    monkeypatch.setattr(TileProcessingStrategy, "find", reject_strategy_lookup)
    kernel = {
        "TileProcessingStrategy": "None", "GlobalSplitU": gsu,
        "GlobalSplitUAlgorithm": "MultipleBuffer", "_GlobalAccumulation": "MultipleBuffer",
        "ProblemType": {"SupportUserArgs": True},
    }

    @contextmanager
    def alloc_tmp(*args, **kwargs):
        yield ContinuousRegister(32, 1)

    writer = SimpleNamespace(
        states=SimpleNamespace(kernel=kernel), allocTmpSgpr=alloc_tmp,
        shiftSrd=lambda ch: Module("gfx950 requires no SRD shift"),
        cmpNamedArgTypeEq=lambda module, value, comment: module.add(
            SCmpEQU32(src0=sgpr("ArgType"), src1=value, comment=comment)),
        sgprs={"GSU": 6, "ArgType": 7, "AddressD": 8, "SrdD": 12},
    )
    module = ".set BufferOOB, 0xffffffff\n.set Srd127_96, 0\n"
    module += str(KernelWriterAssembly.allocPostLoopSrd(writer, "D", kernel))
    observed = run_scalar_kernel(
        writer, module,
        {"GSU": runtime_gsu, "ArgType": arg_type,
         "AddressD+0": 0x9abcdef0, "AddressD+1": 0x12345678},
        ["SrdD+0", "SrdD+1"], tmp_path,
    )
    # Verify both halves so a truncated or incorrectly copied pointer fails.
    assert observed["SrdD+0"] == (0x9abcdef0 if keep_address else 0,) * 64
    assert observed["SrdD+1"] == (0x12345678 if keep_address else 0,) * 64


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("uso", [0, 1])
def test_hybrid_mode_extraction_preserves_uniform_summation_order(tmp_path, mode, uso):
    writer = SimpleNamespace(sgprs={"MagicShiftItersPerTile": 6, "WorkAssignmentMode": 7})
    packed = (mode << 30) | (uso << 29) | 17
    observed = run_scalar_kernel(
        writer, _extract_hybrid_mode(), {"MagicShiftItersPerTile": packed},
        ["WorkAssignmentMode", "MagicShiftItersPerTile"], tmp_path,
    )
    assert observed["WorkAssignmentMode"] == (mode,) * 64
    assert observed["MagicShiftItersPerTile"] == ((uso << 29) | 17,) * 64
