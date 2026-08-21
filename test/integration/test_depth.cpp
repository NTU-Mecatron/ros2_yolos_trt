// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <string>

#include <opencv2/opencv.hpp>

#include "ros2_yolos_trt/adapters/depth_adapter.hpp"
#include "ros2_yolos_trt/conversion/depth_converter.hpp"

namespace {
std::string envOr(const char* name, const std::string& fallback) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : fallback;
}
}  // namespace

class IntegrationFixture : public ::testing::Test {
protected:
  void SetUp() override {
    model_ = envOr("YOLOS_TRT_DEPTH_MODEL", "");
    image_ = envOr("YOLOS_TRT_TEST_IMAGE", "");
    if (model_.empty() || image_.empty()) {
      GTEST_SKIP() << "set YOLOS_TRT_DEPTH_MODEL and YOLOS_TRT_TEST_IMAGE to run";
    }
  }
  std::string model_;
  std::string image_;
  std::string labels_;
};

TEST_F(IntegrationFixture, ProducesMetricDepthAtInputResolution) {
  auto adapter = ros2_yolos_trt::createDepthAdapter();
  ros2_yolos_trt::YolosConfig cfg;
  cfg.model_path = model_;
  cfg.precision = "fp16";
  ASSERT_TRUE(adapter->initialize(cfg)) << "failed to initialize; check the model";

  const cv::Mat frame = cv::imread(image_);
  ASSERT_FALSE(frame.empty()) << "could not read " << image_;

  const auto result = adapter->estimate(frame);
  ASSERT_FALSE(result.depth_metres.empty());

  // Depth must come back at the ORIGINAL image size: this is what catches a
  // broken un-letterbox or resize step.
  EXPECT_EQ(result.depth_metres.cols, frame.cols);
  EXPECT_EQ(result.depth_metres.rows, frame.rows);
  EXPECT_EQ(result.depth_metres.type(), CV_32FC1);

  // Physically plausible metres. A lost metric calibration shows up as
  // near-zero or absurdly large values.
  EXPECT_GT(result.min_metres, 0.0f);
  EXPECT_LT(result.max_metres, 1000.0f);
  EXPECT_GE(result.max_metres, result.min_metres);
}

TEST_F(IntegrationFixture, PublishedImageIsThirtyTwoFloatMetres) {
  auto adapter = ros2_yolos_trt::createDepthAdapter();
  ros2_yolos_trt::YolosConfig cfg;
  cfg.model_path = model_;
  ASSERT_TRUE(adapter->initialize(cfg));

  const cv::Mat frame = cv::imread(image_);
  ASSERT_FALSE(frame.empty());
  const auto result = adapter->estimate(frame);
  ASSERT_FALSE(result.depth_metres.empty());

  const auto msg =
    ros2_yolos_trt::conversion::toDepthImage(result, std_msgs::msg::Header{});

  EXPECT_EQ(msg.encoding, "32FC1");
  EXPECT_EQ(msg.width, static_cast<uint32_t>(frame.cols));
  EXPECT_EQ(msg.height, static_cast<uint32_t>(frame.rows));

  // Spot-check that the published bytes really are the metric values.
  const auto* values = reinterpret_cast<const float*>(msg.data.data());
  const int centre = (frame.rows / 2) * frame.cols + frame.cols / 2;
  EXPECT_FLOAT_EQ(values[centre],
                  result.depth_metres.at<float>(frame.rows / 2, frame.cols / 2));
}

// Second call must reuse the cached engine rather than rebuilding.
TEST_F(IntegrationFixture, SecondInitializeIsFastViaEngineCache) {
  ros2_yolos_trt::YolosConfig cfg;
  cfg.model_path = model_;

  auto first = ros2_yolos_trt::createDepthAdapter();
  ASSERT_TRUE(first->initialize(cfg));
  first->shutdown();

  const auto start = std::chrono::steady_clock::now();
  auto second = ros2_yolos_trt::createDepthAdapter();
  ASSERT_TRUE(second->initialize(cfg));
  const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
    std::chrono::steady_clock::now() - start);

  // A rebuild takes minutes; a cache hit plus warm-up is well under one.
  EXPECT_LT(elapsed.count(), 60);
}
