// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"

using ros2_yolos_trt::DepthResult;
using ros2_yolos_trt::DetectionResult;
using ros2_yolos_trt::YolosConfig;

TEST(Dtos, ConfigDefaultsAreTensorRtAppropriate) {
  YolosConfig cfg;
  // No use_gpu field exists: TensorRT is always GPU. dla_core -1 means GPU.
  EXPECT_EQ(cfg.dla_core, -1);
  EXPECT_EQ(cfg.precision, "fp16");
  EXPECT_FLOAT_EQ(cfg.conf_threshold, 0.4f);
  EXPECT_FLOAT_EQ(cfg.nms_threshold, 0.45f);
}

TEST(Dtos, DetectionResultHoldsBoxAndClass) {
  DetectionResult d;
  d.bbox.x = 5;
  d.bbox.width = 20;
  d.confidence = 0.75f;
  d.class_id = 2;
  d.class_name = "car";
  EXPECT_EQ(d.bbox.x, 5);
  EXPECT_EQ(d.class_name, "car");
}

TEST(Dtos, DepthResultCarriesMetricMap) {
  DepthResult r;
  r.depth_metres = cv::Mat::zeros(4, 6, CV_32FC1);
  r.min_metres = 1.5f;
  r.max_metres = 9.0f;
  EXPECT_EQ(r.depth_metres.type(), CV_32FC1);
  EXPECT_EQ(r.depth_metres.cols, 6);
  EXPECT_FLOAT_EQ(r.min_metres, 1.5f);
}
