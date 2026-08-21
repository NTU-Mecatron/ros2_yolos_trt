// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include "ros2_yolos_trt/conversion/depth_converter.hpp"

using ros2_yolos_trt::DepthResult;
using ros2_yolos_trt::conversion::toDepthImage;

// 32FC1 in metres is the ROS convention; publishing anything else silently
// breaks depth_image_proc, rtabmap and RViz.
TEST(DepthConverter, PublishesThirtyTwoFloatSingleChannel) {
  DepthResult r;
  r.depth_metres = cv::Mat::zeros(6, 8, CV_32FC1);

  std_msgs::msg::Header header;
  header.frame_id = "camera_depth_optical_frame";
  const auto msg = toDepthImage(r, header);

  EXPECT_EQ(msg.encoding, "32FC1");
  EXPECT_EQ(msg.width, 8u);
  EXPECT_EQ(msg.height, 6u);
  EXPECT_EQ(msg.step, 8u * sizeof(float));
  EXPECT_EQ(msg.header.frame_id, "camera_depth_optical_frame");
}

// The model already emits metric depth; the converter must not rescale it.
TEST(DepthConverter, PreservesMetreValuesExactly) {
  DepthResult r;
  r.depth_metres = cv::Mat::zeros(1, 3, CV_32FC1);
  r.depth_metres.at<float>(0, 0) = 0.75f;
  r.depth_metres.at<float>(0, 1) = 4.5f;
  r.depth_metres.at<float>(0, 2) = 12.25f;

  const auto msg = toDepthImage(r, std_msgs::msg::Header{});

  ASSERT_EQ(msg.data.size(), 3u * sizeof(float));
  const auto* values = reinterpret_cast<const float*>(msg.data.data());
  EXPECT_FLOAT_EQ(values[0], 0.75f);
  EXPECT_FLOAT_EQ(values[1], 4.5f);
  EXPECT_FLOAT_EQ(values[2], 12.25f);
}

TEST(DepthConverter, EmptyDepthProducesEmptyImage) {
  const auto msg = toDepthImage(DepthResult{}, std_msgs::msg::Header{});
  EXPECT_EQ(msg.width, 0u);
  EXPECT_EQ(msg.height, 0u);
  EXPECT_TRUE(msg.data.empty());
}
