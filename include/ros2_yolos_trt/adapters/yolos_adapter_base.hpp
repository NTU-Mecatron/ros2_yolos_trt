// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__ADAPTERS__YOLOS_ADAPTER_BASE_HPP_
#define ROS2_YOLOS_TRT__ADAPTERS__YOLOS_ADAPTER_BASE_HPP_

// =============================================================================
// Abstract adapter interfaces and library-agnostic DTOs
// =============================================================================
// No yolos:: type appears in this header. Nodes depend on it, adapters
// implement it, and an upstream version bump therefore touches only adapter
// .cpp files.
// =============================================================================

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "ros2_yolos_trt/visibility_control.hpp"

namespace ros2_yolos_trt {

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

/// @brief Configuration common to every adapter.
///
/// There is deliberately no use_gpu field: TensorRT execution is always on the
/// GPU (or a DLA core). There is deliberately no yolo_version field either:
/// the library detects the version from output tensor shapes.
struct YolosConfig {
  /// .onnx (an engine is built and cached) or .trt/.engine (loaded directly).
  std::string model_path;
  std::string labels_path;
  /// "fp32" or "fp16". Used only when building an engine from .onnx.
  std::string precision{"fp16"};
  /// Empty means "alongside the model file".
  std::string engine_cache_dir;
  /// -1 = GPU, 0/1 = DLA core on Jetson.
  int dla_core{-1};
  float conf_threshold{0.4f};
  float nms_threshold{0.45f};
  int max_detections{300};
};

// -----------------------------------------------------------------------------
// Result DTOs
// -----------------------------------------------------------------------------

struct BoundingBox2D {
  int x{0};
  int y{0};
  int width{0};
  int height{0};
};

struct OrientedBoundingBox2D {
  float center_x{0.0f};
  float center_y{0.0f};
  float width{0.0f};
  float height{0.0f};
  float angle{0.0f};  ///< radians
};

struct KeyPoint2D {
  float x{0.0f};
  float y{0.0f};
  float confidence{0.0f};
  int id{0};
};

struct DetectionResult {
  BoundingBox2D bbox;
  float confidence{0.0f};
  int class_id{0};
  std::string class_name;
};

struct SegmentationResult {
  BoundingBox2D bbox;
  float confidence{0.0f};
  int class_id{0};
  std::string class_name;
  cv::Mat mask;  ///< CV_8UC1 binary mask, original image size
};

struct PoseResult {
  BoundingBox2D bbox;
  float confidence{0.0f};
  int class_id{0};
  std::string class_name;
  std::vector<KeyPoint2D> keypoints;
};

struct OBBResult {
  OrientedBoundingBox2D bbox;
  float confidence{0.0f};
  int class_id{0};
  std::string class_name;
};

struct ClassificationResult {
  int class_id{0};
  float confidence{0.0f};
  std::string class_name;
};

/// @brief Monocular metric depth for one image.
struct DepthResult {
  /// CV_32FC1 at the original image size. Values are METRES, already
  /// calibrated by the exported model; no scaling is applied anywhere.
  cv::Mat depth_metres;
  float min_metres{0.0f};
  float max_metres{0.0f};
};

// -----------------------------------------------------------------------------
// Interfaces
// -----------------------------------------------------------------------------

class ROS2_YOLOS_TRT_PUBLIC IYolosAdapterBase {
public:
  virtual ~IYolosAdapterBase() = default;

  /// @brief Load the model, building an engine from .onnx if required.
  /// @return false on any failure; the caller reports lifecycle FAILURE.
  virtual bool initialize(const YolosConfig& config) = 0;

  [[nodiscard]] virtual bool isInitialized() const noexcept = 0;
  virtual void shutdown() = 0;
  [[nodiscard]] virtual const std::vector<std::string>& getClassNames() const = 0;
};

class ROS2_YOLOS_TRT_PUBLIC IDetectorAdapter : public IYolosAdapterBase {
public:
  virtual std::vector<DetectionResult> detect(
    const cv::Mat& image, float conf_threshold, float nms_threshold) = 0;
  virtual void drawDetections(
    cv::Mat& image, const std::vector<DetectionResult>& detections) = 0;
};

class ROS2_YOLOS_TRT_PUBLIC ISegmentorAdapter : public IYolosAdapterBase {
public:
  virtual std::vector<SegmentationResult> segment(
    const cv::Mat& image, float conf_threshold, float nms_threshold) = 0;
  virtual void drawSegmentations(
    cv::Mat& image, const std::vector<SegmentationResult>& segmentations,
    float mask_alpha = 0.5f) = 0;
};

class ROS2_YOLOS_TRT_PUBLIC IPoseAdapter : public IYolosAdapterBase {
public:
  virtual std::vector<PoseResult> detect(
    const cv::Mat& image, float conf_threshold, float nms_threshold) = 0;
  virtual void drawPoses(
    cv::Mat& image, const std::vector<PoseResult>& poses,
    int kpt_radius = 4, float kpt_threshold = 0.5f) = 0;
};

class ROS2_YOLOS_TRT_PUBLIC IOBBAdapter : public IYolosAdapterBase {
public:
  virtual std::vector<OBBResult> detect(
    const cv::Mat& image, float conf_threshold, float nms_threshold,
    int max_det = 300) = 0;
  virtual void drawDetections(
    cv::Mat& image, const std::vector<OBBResult>& detections) = 0;
};

class ROS2_YOLOS_TRT_PUBLIC IClassifierAdapter : public IYolosAdapterBase {
public:
  virtual ClassificationResult classify(const cv::Mat& image) = 0;
  virtual void drawResult(cv::Mat& image, const ClassificationResult& result) = 0;
};

class ROS2_YOLOS_TRT_PUBLIC IDepthAdapter : public IYolosAdapterBase {
public:
  /// @brief Estimate metric depth. Returns an empty depth_metres on failure.
  virtual DepthResult estimate(const cv::Mat& image) = 0;

  /// @brief Colorized visualization for the debug topic only.
  /// @param min_metres,max_metres clamp range; <= 0 means auto-range.
  virtual cv::Mat colorize(
    const DepthResult& result, float min_metres, float max_metres) = 0;
};

// -----------------------------------------------------------------------------
// Factories
// -----------------------------------------------------------------------------

ROS2_YOLOS_TRT_PUBLIC std::unique_ptr<IDetectorAdapter>   createDetectorAdapter();
ROS2_YOLOS_TRT_PUBLIC std::unique_ptr<ISegmentorAdapter>  createSegmentorAdapter();
ROS2_YOLOS_TRT_PUBLIC std::unique_ptr<IPoseAdapter>       createPoseAdapter();
ROS2_YOLOS_TRT_PUBLIC std::unique_ptr<IOBBAdapter>        createOBBAdapter();
ROS2_YOLOS_TRT_PUBLIC std::unique_ptr<IClassifierAdapter> createClassifierAdapter();
ROS2_YOLOS_TRT_PUBLIC std::unique_ptr<IDepthAdapter>      createDepthAdapter();

}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__ADAPTERS__YOLOS_ADAPTER_BASE_HPP_
