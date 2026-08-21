// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/conversion/classification_converter.hpp"

#include <string>

namespace ros2_yolos_trt {
namespace conversion {

vision_msgs::msg::Classification toClassification(
  const ClassificationResult& result, const std_msgs::msg::Header& header) {
  vision_msgs::msg::Classification msg;
  msg.header = header;

  // The adapter uses -1 to mean "no result"; emitting a hypothesis for it
  // would look like a confident prediction of class -1.
  if (result.class_id < 0) return msg;

  vision_msgs::msg::ObjectHypothesis hyp;
  hyp.class_id = result.class_name.empty() ? std::to_string(result.class_id)
                                           : result.class_name;
  hyp.score = static_cast<double>(result.confidence);
  msg.results.push_back(hyp);
  return msg;
}

}  // namespace conversion
}  // namespace ros2_yolos_trt
