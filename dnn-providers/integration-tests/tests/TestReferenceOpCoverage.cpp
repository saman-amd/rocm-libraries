// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The reference supported-op sets are a commitment: a bundle inside a set gets a
// validation test with no skip path, and one outside it is silently absent from the
// suite. Both halves of that need pinning.

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp>
#include <nlohmann/json.hpp>

#include "RaggedGraphTestUtils.hpp"
#include "SdpaFwdGraphTestUtils.hpp"
#include "harness/bundle/ReferenceOpCoverage.hpp"

using hipdnn_integration_tests::BundleMetadata;
using hipdnn_integration_tests::ReferenceExecutorType;
using hipdnn_integration_tests::bundle::formatUncoveredOps;
using hipdnn_integration_tests::bundle::graphNodeTypes;
using hipdnn_integration_tests::bundle::K_UNREADABLE_GRAPH;
using hipdnn_integration_tests::bundle::NodeAttributes;
using hipdnn_integration_tests::bundle::referenceCoversGraph;
using hipdnn_integration_tests::bundle::referenceSupportedOps;
using hipdnn_integration_tests::bundle::referenceVerificationEnabled;
using hipdnn_integration_tests::bundle::uncoveredNodeTypes;
using hipdnn_integration_tests::test_utils::markFirstTensorRagged;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

// A minimal single-node batchnorm-inference graph, serialized the same way the
// bundle loader does it.
const char* BATCHNORM_GRAPH_JSON = R"({"nodes": [{"inputs": {"x_tensor_uid": 0,
    "mean_tensor_uid": 1, "inv_variance_tensor_uid": 2, "scale_tensor_uid": 3,
    "bias_tensor_uid": 4}, "outputs": {"y_tensor_uid": 5},
    "type": "BatchnormInferenceAttributes", "compute_data_type": "float", "name": ""}],
    "tensors": [
    {"name": "", "uid": 0, "strides": [60, 20, 5, 1], "dims": [2, 3, 4, 5], "data_type": "float", "virtual": false},
    {"name": "", "uid": 1, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], "data_type": "float", "virtual": false},
    {"name": "", "uid": 2, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], "data_type": "float", "virtual": false},
    {"name": "", "uid": 3, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], "data_type": "float", "virtual": false},
    {"name": "", "uid": 4, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], "data_type": "float", "virtual": false},
    {"name": "", "uid": 5, "strides": [60, 20, 5, 1], "dims": [2, 3, 4, 5], "data_type": "float", "virtual": false}],
    "io_data_type": "float", "compute_data_type": "float",
    "intermediate_data_type": "float", "name": ""})";

flatbuffers::DetachedBuffer buildBatchnormGraph()
{
    flatbuffers::FlatBufferBuilder builder;
    const auto json = nlohmann::json::parse(BATCHNORM_GRAPH_JSON);
    auto offset = hipdnn_flatbuffers_sdk::json::to<hipdnn_flatbuffers_sdk::data_objects::Graph>(
        builder, json);
    builder.Finish(offset);
    return builder.Release();
}

flatbuffers::FlatBufferBuilder
    buildSdpaGraph(hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT attrs = {})
{
    const std::vector<int64_t> dims = {1, 2, 64, 16};
    return hipdnn_integration_tests::test_utils::createSdpaFwdGraph(
        0,
        1,
        2,
        3,
        dims,
        dims,
        dims,
        dims,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        attrs);
}

} // namespace

// ---------------------------------------------------------------------------
// The sets themselves
// ---------------------------------------------------------------------------

TEST(TestReferenceOpCoverage, BothReferenceSetsAreNonEmpty)
{
    EXPECT_FALSE(referenceSupportedOps(ReferenceExecutorType::CPU).empty());
    EXPECT_FALSE(referenceSupportedOps(ReferenceExecutorType::GPU).empty());
}

// The two references cover different ops on purpose — the GPU one dispatches
// through a signature-keyed plan registry and grows only as builders are written.
// If these ever became identical the split would be pointless, so it is worth
// noticing.
TEST(TestReferenceOpCoverage, SetsAreIndependent)
{
    EXPECT_NE(referenceSupportedOps(ReferenceExecutorType::CPU),
              referenceSupportedOps(ReferenceExecutorType::GPU));
}

// ---------------------------------------------------------------------------
// Graph inspection
// ---------------------------------------------------------------------------

TEST(TestReferenceOpCoverage, NodeTypesAreReadFromTheGraph)
{
    const auto graph = buildBatchnormGraph();
    const auto types = graphNodeTypes(graph.data(), graph.size());

    ASSERT_TRUE(types.has_value());
    ASSERT_EQ(types->size(), 1u);
    EXPECT_EQ(*types->begin(), NodeAttributes::BatchnormInferenceAttributes);
}

// An unreadable buffer must not be treated as "covered by everything" — that would
// register a validation test for a bundle nobody can run.
TEST(TestReferenceOpCoverage, UnreadableGraphIsNotCovered)
{
    const std::vector<uint8_t> garbage(64, 0xAB);

    EXPECT_FALSE(graphNodeTypes(garbage.data(), garbage.size()).has_value());
    EXPECT_FALSE(referenceCoversGraph(ReferenceExecutorType::CPU, garbage.data(), garbage.size()));
    EXPECT_FALSE(referenceCoversGraph(ReferenceExecutorType::GPU, garbage.data(), garbage.size()));
}

// "Not covered" and "nothing is uncovered" must not both be true of one graph: the
// registration log prints an exclusion count next to the ops responsible for it, so
// an unreadable graph that named no ops would report a gap with no reason attached.
TEST(TestReferenceOpCoverage, UnreadableGraphNamesItselfAsTheReason)
{
    const std::vector<uint8_t> garbage(64, 0xAB);

    const auto uncovered
        = uncoveredNodeTypes(ReferenceExecutorType::CPU, garbage.data(), garbage.size());
    ASSERT_EQ(uncovered.size(), 1u);
    EXPECT_EQ(uncovered.front(), K_UNREADABLE_GRAPH);
}

// ---------------------------------------------------------------------------
// Coverage decision
// ---------------------------------------------------------------------------

TEST(TestReferenceOpCoverage, CpuCoversBatchnormInference)
{
    const auto graph = buildBatchnormGraph();
    EXPECT_TRUE(referenceCoversGraph(ReferenceExecutorType::CPU, graph.data(), graph.size()));
    EXPECT_TRUE(uncoveredNodeTypes(ReferenceExecutorType::CPU, graph.data(), graph.size()).empty());
}

// The GPU reference has no batchnorm plan builder, so bundles using it are absent
// from the GPU validation suite rather than skipped inside it.
TEST(TestReferenceOpCoverage, DeviceReferenceDoesNotCoverBatchnormInference)
{
    const auto graph = buildBatchnormGraph();
    EXPECT_FALSE(referenceCoversGraph(ReferenceExecutorType::GPU, graph.data(), graph.size()));

    const auto uncovered
        = uncoveredNodeTypes(ReferenceExecutorType::GPU, graph.data(), graph.size());
    ASSERT_EQ(uncovered.size(), 1u);
    EXPECT_EQ(uncovered.front(), "BatchnormInferenceAttributes");
}

// Coverage is decided by op type alone. Graph features a reference cannot handle
// are declared per bundle through reference_verification, which the harness then
// checks against isApplicable() rather than silently dropping the bundle.
TEST(TestReferenceOpCoverage, RaggedAndSeqLenGraphsAreCoveredByOpType)
{
    auto dense = buildSdpaGraph();
    auto ragged = markFirstTensorRagged(dense.GetBufferPointer());

    hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT seqLenAttrs;
    seqLenAttrs.seq_len_kv_tensor_uid = 4;
    auto seqLen = buildSdpaGraph(seqLenAttrs);

    EXPECT_TRUE(referenceCoversGraph(ReferenceExecutorType::GPU, ragged.data(), ragged.size()));
    EXPECT_TRUE(referenceCoversGraph(
        ReferenceExecutorType::GPU, seqLen.GetBufferPointer(), seqLen.GetSize()));
}

// ---------------------------------------------------------------------------
// reference_verification gate
// ---------------------------------------------------------------------------

TEST(TestReferenceOpCoverage, ReferenceVerificationEnabledByDefault)
{
    const BundleMetadata metadata;
    EXPECT_TRUE(referenceVerificationEnabled(ReferenceExecutorType::CPU, metadata));
    EXPECT_TRUE(referenceVerificationEnabled(ReferenceExecutorType::GPU, metadata));
}

TEST(TestReferenceOpCoverage, ReferenceVerificationFlagsAreIndependent)
{
    BundleMetadata gpuDisabled;
    gpuDisabled.referenceVerification.gpu = false;
    EXPECT_TRUE(referenceVerificationEnabled(ReferenceExecutorType::CPU, gpuDisabled));
    EXPECT_FALSE(referenceVerificationEnabled(ReferenceExecutorType::GPU, gpuDisabled));

    BundleMetadata cpuDisabled;
    cpuDisabled.referenceVerification.cpu = false;
    EXPECT_FALSE(referenceVerificationEnabled(ReferenceExecutorType::CPU, cpuDisabled));
    EXPECT_TRUE(referenceVerificationEnabled(ReferenceExecutorType::GPU, cpuDisabled));
}

// ---------------------------------------------------------------------------
// Registration diagnostic
//
// The exclusion tally alone says a gap exists without saying which op to
// implement to close it, which is what made uncoveredNodeTypes() dead code.
// ---------------------------------------------------------------------------

TEST(TestReferenceOpCoverage, NoExclusionsAddsNothingToTheSummary)
{
    EXPECT_EQ(formatUncoveredOps({}), "");
}

TEST(TestReferenceOpCoverage, ExcludedOpsAreNamedAndSeparated)
{
    EXPECT_EQ(formatUncoveredOps({"BatchnormInferenceAttributes"}),
              " (BatchnormInferenceAttributes)");
    EXPECT_EQ(formatUncoveredOps({"ReductionAttributes", "ConvolutionBwdDataAttributes"}),
              " (ConvolutionBwdDataAttributes, ReductionAttributes)");
}

// NOLINTEND(readability-identifier-naming)
