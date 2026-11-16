// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/labels/label_map.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>

namespace ros2_yolos_trt {
namespace {

std::string trim(const std::string& s) {
  const size_t first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const size_t last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

bool parseColor(const std::string& token, cv::Scalar& bgr) {
  if (token.size() != 7 || token[0] != '#') return false;
  if (!std::all_of(token.begin() + 1, token.end(), [](unsigned char c) {
        return std::isxdigit(c) != 0;
      })) {
    return false;
  }
  const int r = std::stoi(token.substr(1, 2), nullptr, 16);
  const int g = std::stoi(token.substr(3, 2), nullptr, 16);
  const int b = std::stoi(token.substr(5, 2), nullptr, 16);
  bgr = cv::Scalar(b, g, r);
  return true;
}

}  // namespace

bool loadLabelMap(const std::string& path, bool require_colors, LabelMap& out,
                  std::string& error) {
  out = LabelMap{};
  error.clear();

  std::ifstream file(path);
  if (!file.is_open()) {
    error = "cannot open labels file: " + path;
    return false;
  }

  std::string raw;
  int line_no = 0;
  while (std::getline(file, raw)) {
    ++line_no;
    const std::string line = trim(raw);
    if (line.empty()) continue;

    std::string name = line;
    cv::Scalar color(0, 255, 0);
    bool has_color = false;

    const size_t sep = line.find_last_of(" \t");
    if (sep != std::string::npos) {
      cv::Scalar parsed;
      if (parseColor(trim(line.substr(sep + 1)), parsed)) {
        color = parsed;
        has_color = true;
        name = trim(line.substr(0, sep));
      }
    }

    if (require_colors && !has_color) {
      error = path + ":" + std::to_string(line_no) +
              ": expected '<name> #rrggbb', got '" + line + "'";
      return false;
    }
    if (name.empty()) {
      error = path + ":" + std::to_string(line_no) + ": empty class name";
      return false;
    }
    out.names.push_back(std::move(name));
    out.colors_bgr.push_back(color);
  }

  if (out.names.empty()) {
    error = "labels file has no classes: " + path;
    return false;
  }
  return true;
}

}  // namespace ros2_yolos_trt
