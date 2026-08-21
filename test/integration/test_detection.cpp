// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <string>

#include <opencv2/opencv.hpp>

#include "ros2_yolos_trt/adapters/detector_adapter.hpp"

namespace {
std::string envOr(const char* name, const std::string& fallback) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : fallback;
}
}  // namespace

class IntegrationFixture : public ::testing::Test {
protected:
  void SetUp() override {
    model_ = envOr("YOLOS_TRT_DETECTION_MODEL", "");
    image_ = envOr("YOLOS_TRT_TEST_IMAGE", "");
    labels_ = envOr("YOLOS_TRT_TEST_LABELS", "");
    if (model_.empty() || image_.empty()) {
      GTEST_SKIP() << "set YOLOS_TRT_DETECTION_MODEL and YOLOS_TRT_TEST_IMAGE to run";
    }
  }
  std::string model_;
  std::string image_;
  std::string labels_;
};

TEST_F(IntegrationFixture, DetectsWithinImageBounds) {
  auto adapter = ros2_yolos_trt::createDetectorAdapter();
  ros2_yolos_trt::YolosConfig cfg;
  cfg.model_path = model_;
  cfg.labels_path = labels_;
  ASSERT_TRUE(adapter->initialize(cfg)) << "failed to initialize; check the model";
  EXPECT_FALSE(adapter->getClassNames().empty()) << "labels did not load";

  const cv::Mat frame = cv::imread(image_);
  ASSERT_FALSE(frame.empty()) << "could not read " << image_;

  const auto dets = adapter->detect(frame, 0.4f, 0.45f);
  for (const auto& d : dets) {
    EXPECT_GT(d.bbox.width, 0);
    EXPECT_GT(d.bbox.height, 0);
    EXPECT_GE(d.bbox.x, 0);
    EXPECT_GE(d.bbox.y, 0);
    EXPECT_LE(d.bbox.x + d.bbox.width, frame.cols);
    EXPECT_LE(d.bbox.y + d.bbox.height, frame.rows);
    EXPECT_GE(d.confidence, 0.0f);
    EXPECT_LE(d.confidence, 1.0f);
  }
}
