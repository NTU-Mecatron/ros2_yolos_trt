// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <string>

#include <opencv2/opencv.hpp>

#include "ros2_yolos_trt/adapters/segmentor_adapter.hpp"

namespace {
std::string envOr(const char* name, const std::string& fallback) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : fallback;
}
}  // namespace

class IntegrationFixture : public ::testing::Test {
protected:
  void SetUp() override {
    model_ = envOr("YOLOS_TRT_SEGMENTATION_MODEL", "");
    image_ = envOr("YOLOS_TRT_TEST_IMAGE", "");
    labels_ = envOr("YOLOS_TRT_TEST_LABELS", "");
    if (model_.empty() || image_.empty()) {
      GTEST_SKIP() << "set YOLOS_TRT_SEGMENTATION_MODEL and YOLOS_TRT_TEST_IMAGE to run";
    }
  }
  std::string model_;
  std::string image_;
  std::string labels_;
};

TEST_F(IntegrationFixture, MasksMatchTheInputImage) {
  auto adapter = ros2_yolos_trt::createSegmentorAdapter();
  ros2_yolos_trt::YolosConfig cfg;
  cfg.model_path = model_;
  cfg.labels_path = labels_;
  ASSERT_TRUE(adapter->initialize(cfg));

  const cv::Mat frame = cv::imread(image_);
  ASSERT_FALSE(frame.empty());

  const auto segs = adapter->segment(frame, 0.4f, 0.45f);
  for (const auto& s : segs) {
    ASSERT_FALSE(s.mask.empty()) << "segmentation returned an empty mask";
    EXPECT_EQ(s.mask.type(), CV_8UC1);
    EXPECT_EQ(s.mask.cols, frame.cols);
    EXPECT_EQ(s.mask.rows, frame.rows);
  }
}
