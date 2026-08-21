// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/nodes/detector_node.hpp"

#include <chrono>
#include <memory>

#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include "ros2_yolos_trt/conversion/detection_converter.hpp"

namespace ros2_yolos_trt {

YolosDetectorNode::YolosDetectorNode(const rclcpp::NodeOptions& options)
: YolosLifecycleNodeBase("yolos_trt_detector", options) {
  declareCommonParameters();
}

YolosDetectorNode::CallbackReturn YolosDetectorNode::on_configure(
  const rclcpp_lifecycle::State&) {
  config_ = loadCommonConfig();
  if (config_.model_path.empty()) {
    RCLCPP_ERROR(get_logger(), "model_path parameter is required");
    return CallbackReturn::FAILURE;
  }

  adapter_ = createDetectorAdapter();

  // Building an engine from .onnx happens here and can take minutes; doing it
  // in on_configure is what lets the lifecycle state report the truth.
  RCLCPP_INFO(get_logger(), "loading model %s", config_.model_path.c_str());
  if (!adapter_->initialize(config_)) {
    RCLCPP_ERROR(get_logger(), "failed to initialize adapter");
    adapter_.reset();
    return CallbackReturn::FAILURE;
  }

  det_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>(
    "~/detections", 10);
  if (publishDebugImage()) {
    debug_pub_ = create_publisher<sensor_msgs::msg::Image>("~/debug_image", 10);
  }
  createTimingPublisher();

  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = inference_cb_group_;
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    "image", rclcpp::SensorDataQoS(),
    [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
      imageCallback(msg);
    },
    sub_options);

  RCLCPP_INFO(get_logger(), "configured with %zu classes", adapter_->getClassNames().size());
  return CallbackReturn::SUCCESS;
}

YolosDetectorNode::CallbackReturn YolosDetectorNode::on_activate(
  const rclcpp_lifecycle::State& state) {
  LifecycleNode::on_activate(state);
  return CallbackReturn::SUCCESS;
}

YolosDetectorNode::CallbackReturn YolosDetectorNode::on_deactivate(
  const rclcpp_lifecycle::State& state) {
  LifecycleNode::on_deactivate(state);
  return CallbackReturn::SUCCESS;
}

YolosDetectorNode::CallbackReturn YolosDetectorNode::on_cleanup(
  const rclcpp_lifecycle::State&) {
  image_sub_.reset();
  det_pub_.reset();
  debug_pub_.reset();
  timing_pub_.reset();
  if (adapter_) adapter_->shutdown();
  adapter_.reset();
  return CallbackReturn::SUCCESS;
}

YolosDetectorNode::CallbackReturn YolosDetectorNode::on_shutdown(
  const rclcpp_lifecycle::State& state) {
  return on_cleanup(state);
}

void YolosDetectorNode::imageCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
  if (!adapter_ || !adapter_->isInitialized()) return;
  if (!det_pub_ || !det_pub_->is_activated()) return;

  const auto t_start = std::chrono::steady_clock::now();

  cv::Mat frame;
  try {
    frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
  } catch (const cv_bridge::Exception& e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
    return;
  }

  const auto t_infer_start = std::chrono::steady_clock::now();
  auto detections =
    adapter_->detect(frame, config_.conf_threshold, config_.nms_threshold);
  const auto t_infer_end = std::chrono::steady_clock::now();

  det_pub_->publish(conversion::toDetection2DArray(detections, msg->header));

  if (debug_pub_ && debug_pub_->is_activated()) {
    adapter_->drawDetections(frame, detections);
    debug_pub_->publish(
      *cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg());
  }

  const auto t_end = std::chrono::steady_clock::now();
  using ms = std::chrono::duration<double, std::milli>;
  publishTimingSample(ms(t_infer_end - t_infer_start).count(),
                      ms(t_end - t_start).count());
}

}  // namespace ros2_yolos_trt

RCLCPP_COMPONENTS_REGISTER_NODE(ros2_yolos_trt::YolosDetectorNode)
