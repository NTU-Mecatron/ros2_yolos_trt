// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <string>

#include <opencv2/opencv.hpp>

#include "ros2_yolos_trt/adapters/obb_adapter.hpp"

#include <cmath>

namespace {
std::string envOr(const char* name, const std::string& fallback) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : fallback;
}
}  // namespace

class IntegrationFixture : public ::testing::Test {
protected:
  void SetUp() override {
    model_ = envOr("YOLOS_TRT_OBB_MODEL", "");
    image_ = envOr("YOLOS_TRT_TEST_IMAGE", "");
    labels_ = envOr("YOLOS_TRT_TEST_LABELS", "");
    if (model_.empty() || image_.empty()) {
      GTEST_SKIP() << "set YOLOS_TRT_OBB_MODEL and YOLOS_TRT_TEST_IMAGE to run";
    }
  }
  std::string model_;
  std::string image_;
  std::string labels_;
};

TEST_F(IntegrationFixture, OrientedBoxesHaveSaneGeometry) {
  auto adapter = ros2_yolos_trt::createOBBAdapter();
  ros2_yolos_trt::YolosConfig cfg;
  cfg.model_path = model_;
  cfg.labels_path = labels_;
  ASSERT_TRUE(adapter->initialize(cfg));

  const cv::Mat frame = cv::imread(image_);
  ASSERT_FALSE(frame.empty());

  const auto dets = adapter->detect(frame, 0.4f, 0.45f, 300);
  for (const auto& d : dets) {
    EXPECT_GT(d.bbox.width, 0.0f);
    EXPECT_GT(d.bbox.height, 0.0f);
    EXPECT_GE(d.bbox.angle, -static_cast<float>(M_PI) - 1e-3f);
    EXPECT_LE(d.bbox.angle, static_cast<float>(M_PI) + 1e-3f);
  }
}
