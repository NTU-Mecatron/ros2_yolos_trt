// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__LABELS__LABEL_MAP_HPP_
#define ROS2_YOLOS_TRT__LABELS__LABEL_MAP_HPP_

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "ros2_yolos_trt/visibility_control.hpp"

namespace ros2_yolos_trt {

/// @brief Class names and display colors parsed from a labels file.
///
/// File format, one class per line, line N = class id N (blank lines skipped):
///   mixed-pole #611717
///   red-pole   #c90404
/// The trailing `#rrggbb` token is the display color. Names may contain spaces.
struct LabelMap {
  std::vector<std::string> names;
  /// Same length as names. BGR, ready for OpenCV. Lines without a color get
  /// green when colors are optional.
  std::vector<cv::Scalar> colors_bgr;
};

/// @brief Parse a labels file.
/// @param require_colors true: every line must end in a valid `#rrggbb`.
///                       false: a line without one is taken as a plain name.
/// @return false and fills @p error on failure.
ROS2_YOLOS_TRT_PUBLIC bool loadLabelMap(
  const std::string& path, bool require_colors, LabelMap& out,
  std::string& error);

}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__LABELS__LABEL_MAP_HPP_
