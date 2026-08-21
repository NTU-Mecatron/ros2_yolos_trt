// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <string>

#include <opencv2/opencv.hpp>

#include "ros2_yolos_trt/adapters/pose_adapter.hpp"

namespace {
std::string envOr(const char* name, const std::string& fallback) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : fallback;
}
}  // namespace

class IntegrationFixture : public ::testing::Test {
protected:
  void SetUp() override {
    model_ = envOr("YOLOS_TRT_POSE_MODEL", "");
    image_ = envOr("YOLOS_TRT_TEST_IMAGE", "");
    labels_ = envOr("YOLOS_TRT_TEST_LABELS", "");
    if (model_.empty() || image_.empty()) {
      GTEST_SKIP() << "set YOLOS_TRT_POSE_MODEL and YOLOS_TRT_TEST_IMAGE to run";
    }
  }
  std::string model_;
  std::string image_;
  std::string labels_;
};

// Regression guard for the keypoint plumbing: the reference ONNX package
// computed keypoints and then discarded them.
TEST_F(IntegrationFixture, KeypointsArePopulatedAndInBounds) {
  auto adapter = ros2_yolos_trt::createPoseAdapter();
  ros2_yolos_trt::YolosConfig cfg;
  cfg.model_path = model_;
  cfg.labels_path = labels_;
  ASSERT_TRUE(adapter->initialize(cfg));

  const cv::Mat frame = cv::imread(image_);
  ASSERT_FALSE(frame.empty());

  const auto poses = adapter->detect(frame, 0.4f, 0.45f);
  for (const auto& p : poses) {
    EXPECT_FALSE(p.keypoints.empty()) << "pose returned no keypoints";
    int expected_id = 0;
    for (const auto& kp : p.keypoints) {
      EXPECT_EQ(kp.id, expected_id++) << "keypoint ids must be positional";
      EXPECT_GE(kp.x, -1.0f);
      EXPECT_GE(kp.y, -1.0f);
      EXPECT_LE(kp.x, static_cast<float>(frame.cols));
      EXPECT_LE(kp.y, static_cast<float>(frame.rows));
    }
  }
}
