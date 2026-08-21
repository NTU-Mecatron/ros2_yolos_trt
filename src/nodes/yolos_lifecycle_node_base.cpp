// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/nodes/yolos_lifecycle_node_base.hpp"

#include <vector>

namespace ros2_yolos_trt {

YolosLifecycleNodeBase::YolosLifecycleNodeBase(const std::string& node_name,
                                               const rclcpp::NodeOptions& options)
: rclcpp_lifecycle::LifecycleNode(node_name, options) {
  inference_cb_group_ =
    create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
}

void YolosLifecycleNodeBase::declareCommonParameters() {
  declare_parameter<std::string>("model_path", "");
  declare_parameter<std::string>("labels_path", "");
  declare_parameter<std::string>("precision", "fp16");
  declare_parameter<std::string>("engine_cache_dir", "");
  declare_parameter<int>("dla_core", -1);
  declare_parameter<double>("conf_threshold", 0.4);
  declare_parameter<double>("nms_threshold", 0.45);
  declare_parameter<int>("max_detections", 300);
  declare_parameter<double>("depth_min", 0.0);
  declare_parameter<double>("depth_max", 0.0);
  declare_parameter<bool>("publish_debug_image", false);
  declare_parameter<bool>("publish_timing", false);
}

YolosConfig YolosLifecycleNodeBase::loadCommonConfig() const {
  YolosConfig cfg;
  cfg.model_path = get_parameter("model_path").as_string();
  cfg.labels_path = get_parameter("labels_path").as_string();
  cfg.precision = get_parameter("precision").as_string();
  cfg.engine_cache_dir = get_parameter("engine_cache_dir").as_string();
  cfg.dla_core = static_cast<int>(get_parameter("dla_core").as_int());
  cfg.conf_threshold =
    static_cast<float>(get_parameter("conf_threshold").as_double());
  cfg.nms_threshold =
    static_cast<float>(get_parameter("nms_threshold").as_double());
  cfg.max_detections =
    static_cast<int>(get_parameter("max_detections").as_int());
  return cfg;
}

bool YolosLifecycleNodeBase::publishDebugImage() const {
  return get_parameter("publish_debug_image").as_bool();
}

bool YolosLifecycleNodeBase::publishTimingEnabled() const {
  return get_parameter("publish_timing").as_bool();
}

float YolosLifecycleNodeBase::depthMin() const {
  return static_cast<float>(get_parameter("depth_min").as_double());
}

float YolosLifecycleNodeBase::depthMax() const {
  return static_cast<float>(get_parameter("depth_max").as_double());
}

int YolosLifecycleNodeBase::maxDetections() const {
  return static_cast<int>(get_parameter("max_detections").as_int());
}

void YolosLifecycleNodeBase::createTimingPublisher() {
  timing_pub_ =
    create_publisher<std_msgs::msg::Float64MultiArray>("~/timing", 10);
}

void YolosLifecycleNodeBase::publishTimingSample(double inference_ms,
                                                 double total_ms) {
  if (!publishTimingEnabled() || !timing_pub_ || !timing_pub_->is_activated()) {
    return;
  }
  std_msgs::msg::Float64MultiArray msg;
  msg.data = {inference_ms, total_ms};
  timing_pub_->publish(msg);
}

}  // namespace ros2_yolos_trt
