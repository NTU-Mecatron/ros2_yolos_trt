// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include "ros2_yolos_trt/conversion/pose_converter.hpp"

using ros2_yolos_trt::KeyPoint2D;
using ros2_yolos_trt::PoseResult;
using ros2_yolos_trt::conversion::poseToDetection2DArray;
using ros2_yolos_trt::conversion::toKeyPoint2DArrays;

namespace {
PoseResult makePose() {
  PoseResult p;
  p.bbox.x = 10;
  p.bbox.y = 20;
  p.bbox.width = 30;
  p.bbox.height = 60;
  p.confidence = 0.9f;
  p.class_id = 0;
  p.class_name = "person";
  p.keypoints = {
    KeyPoint2D{15.0f, 25.0f, 0.95f, 0},
    KeyPoint2D{20.0f, 35.0f, 0.40f, 1},
  };
  return p;
}
}  // namespace

TEST(PoseConverter, BoxesConvertToCentreSize) {
  const auto msg = poseToDetection2DArray({makePose()}, std_msgs::msg::Header{});
  ASSERT_EQ(msg.detections.size(), 1u);
  EXPECT_DOUBLE_EQ(msg.detections[0].bbox.center.position.x, 25.0);
  EXPECT_DOUBLE_EQ(msg.detections[0].bbox.center.position.y, 50.0);
}

// The whole point of this converter: keypoints must survive to a message.
TEST(PoseConverter, KeypointsArePreservedPerInstance) {
  const auto arrays = toKeyPoint2DArrays({makePose()}, std_msgs::msg::Header{});
  ASSERT_EQ(arrays.size(), 1u);
  ASSERT_EQ(arrays[0].keypoints.size(), 2u);
  EXPECT_DOUBLE_EQ(arrays[0].keypoints[0].x, 15.0);
  EXPECT_DOUBLE_EQ(arrays[0].keypoints[0].y, 25.0);
  EXPECT_EQ(arrays[0].keypoints[1].id, 1);
  EXPECT_NEAR(arrays[0].keypoints[1].confidence, 0.40, 1e-6);
}

TEST(PoseConverter, KeypointArrayCarriesInstanceBoxAndClass) {
  const auto arrays = toKeyPoint2DArrays({makePose()}, std_msgs::msg::Header{});
  ASSERT_EQ(arrays.size(), 1u);
  EXPECT_DOUBLE_EQ(arrays[0].bounding_box_center.x, 25.0);
  EXPECT_DOUBLE_EQ(arrays[0].bounding_box_size_x, 30.0);
  EXPECT_EQ(arrays[0].class_id, 0);
  EXPECT_NEAR(arrays[0].confidence, 0.9, 1e-6);
}

TEST(PoseConverter, EmptyInputProducesNoArrays) {
  EXPECT_TRUE(toKeyPoint2DArrays({}, std_msgs::msg::Header{}).empty());
}
