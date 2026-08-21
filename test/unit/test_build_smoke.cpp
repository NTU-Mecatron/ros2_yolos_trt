// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include <NvInferVersion.h>

#include "yolos/core/cuda_preprocessing.hpp"
#include "yolos/core/version.hpp"

// TensorRT 11.x removed API this library depends on; the build must pin 10.x.
TEST(BuildSmoke, TensorRtMajorVersionIsTen) {
  EXPECT_EQ(NV_TENSORRT_MAJOR, 10);
}

TEST(BuildSmoke, UpstreamVersionEnumIsReachable) {
  EXPECT_EQ(yolos::version::toString(yolos::YOLOVersion::V26), "V26");
}

TEST(BuildSmoke, CudaPreprocessDeclarationLinks) {
  // Only checks the symbol resolves at link time; no CUDA call is made.
  auto* fn = &yolos::cuda::letterboxPreprocess;
  EXPECT_NE(fn, nullptr);
}
