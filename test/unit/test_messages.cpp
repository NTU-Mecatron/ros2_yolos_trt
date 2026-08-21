// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include "ros2_yolos_trt/msg/key_point2_d.hpp"
#include "ros2_yolos_trt/msg/key_point2_d_array.hpp"
#include "ros2_yolos_trt/msg/obb_detection2_d.hpp"
#include "ros2_yolos_trt/msg/obb_detection2_d_array.hpp"
#include "ros2_yolos_trt/msg/oriented_bounding_box2_d.hpp"

TEST(Messages, OrientedBoundingBoxHoldsGeometry) {
  ros2_yolos_trt::msg::OrientedBoundingBox2D obb;
  obb.center.x = 10.0;
  obb.center.y = 20.0;
  obb.size_x = 30.0;
  obb.size_y = 40.0;
  obb.theta = 0.5;
  EXPECT_DOUBLE_EQ(obb.center.x, 10.0);
  EXPECT_DOUBLE_EQ(obb.theta, 0.5);
}

TEST(Messages, OBBDetectionArrayNestsDetections) {
  ros2_yolos_trt::msg::OBBDetection2DArray arr;
  ros2_yolos_trt::msg::OBBDetection2D det;
  det.class_id = 3;
  det.class_name = "ship";
  det.score = 0.9;
  arr.detections.push_back(det);
  ASSERT_EQ(arr.detections.size(), 1u);
  EXPECT_EQ(arr.detections[0].class_name, "ship");
}

TEST(Messages, KeyPointArrayNestsKeypoints) {
  ros2_yolos_trt::msg::KeyPoint2DArray arr;
  ros2_yolos_trt::msg::KeyPoint2D kp;
  kp.x = 1.0;
  kp.y = 2.0;
  kp.confidence = 0.8;
  kp.id = 5;
  arr.keypoints.push_back(kp);
  arr.class_id = 0;
  ASSERT_EQ(arr.keypoints.size(), 1u);
  EXPECT_EQ(arr.keypoints[0].id, 5);
}
