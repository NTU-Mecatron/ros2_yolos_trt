// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__NODES__POSE_NODE_HPP_
#define ROS2_YOLOS_TRT__NODES__POSE_NODE_HPP_

#include <memory>

#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "ros2_yolos_trt/msg/key_point2_d_array.hpp"

#include "ros2_yolos_trt/adapters/pose_adapter.hpp"
#include "ros2_yolos_trt/nodes/yolos_lifecycle_node_base.hpp"

namespace ros2_yolos_trt {

class ROS2_YOLOS_TRT_PUBLIC YolosPoseNode : public YolosLifecycleNodeBase {
public:
  explicit YolosPoseNode(const rclcpp::NodeOptions& options);

  CallbackReturn on_configure(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State& state) override;

private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

  std::unique_ptr<IPoseAdapter> adapter_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp_lifecycle::LifecyclePublisher<vision_msgs::msg::Detection2DArray>::SharedPtr
    det_pub_;
  rclcpp_lifecycle::LifecyclePublisher<msg::KeyPoint2DArray>::SharedPtr kpt_pub_;
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr
    debug_pub_;
  YolosConfig config_;
};

}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__NODES__POSE_NODE_HPP_
