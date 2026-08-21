// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "ros2_yolos_trt/engine/engine_cache.hpp"

namespace fs = std::filesystem;
using ros2_yolos_trt::engine::cacheKey;
using ros2_yolos_trt::engine::isEnginePath;
using ros2_yolos_trt::engine::resolveEngine;

namespace {
fs::path makeTempOnnx(const std::string& name, const std::string& contents) {
  const fs::path p = fs::temp_directory_path() / name;
  std::ofstream f(p, std::ios::binary);
  f << contents;
  f.close();
  return p;
}
}  // namespace

TEST(EngineCache, RecognisesEngineExtensions) {
  EXPECT_TRUE(isEnginePath("/models/a.trt"));
  EXPECT_TRUE(isEnginePath("/models/a.engine"));
  EXPECT_FALSE(isEnginePath("/models/a.onnx"));
  EXPECT_FALSE(isEnginePath("/models/a"));
}

TEST(EngineCache, KeyIsStableForIdenticalInputs) {
  const auto onnx = makeTempOnnx("ec_stable.onnx", "abc");
  EXPECT_EQ(cacheKey(onnx.string(), "fp16", -1),
            cacheKey(onnx.string(), "fp16", -1));
}

TEST(EngineCache, KeyChangesWithPrecision) {
  const auto onnx = makeTempOnnx("ec_prec.onnx", "abc");
  EXPECT_NE(cacheKey(onnx.string(), "fp16", -1),
            cacheKey(onnx.string(), "fp32", -1));
}

TEST(EngineCache, KeyChangesWithDlaCore) {
  const auto onnx = makeTempOnnx("ec_dla.onnx", "abc");
  EXPECT_NE(cacheKey(onnx.string(), "fp16", -1),
            cacheKey(onnx.string(), "fp16", 0));
}

// An engine built from different model bytes must not be reused, or the node
// silently runs the wrong network.
TEST(EngineCache, KeyChangesWithModelContents) {
  const auto a = makeTempOnnx("ec_a.onnx", "aaaa");
  const auto b = makeTempOnnx("ec_b.onnx", "bbbbbbbb");
  EXPECT_NE(cacheKey(a.string(), "fp16", -1), cacheKey(b.string(), "fp16", -1));
}

TEST(EngineCache, PrebuiltEnginePathIsReturnedUnchanged) {
  std::string err;
  const auto out = resolveEngine("/models/x.trt", "fp16", -1, "", err);
  EXPECT_EQ(out.string(), "/models/x.trt");
  EXPECT_TRUE(err.empty());
}

TEST(EngineCache, MissingModelReportsAnError) {
  std::string err;
  const auto out = resolveEngine("/nope/missing.onnx", "fp16", -1, "", err);
  EXPECT_TRUE(out.empty());
  EXPECT_NE(err.find("not found"), std::string::npos);
}

TEST(EngineCache, UnsupportedExtensionReportsAnError) {
  const auto bad = makeTempOnnx("ec_bad.pt", "x");
  std::string err;
  const auto out = resolveEngine(bad.string(), "fp16", -1, "", err);
  EXPECT_TRUE(out.empty());
  EXPECT_NE(err.find(".onnx"), std::string::npos);
}
