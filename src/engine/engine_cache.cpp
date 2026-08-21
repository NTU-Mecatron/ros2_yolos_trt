// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/engine/engine_cache.hpp"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

namespace fs = std::filesystem;

namespace ros2_yolos_trt {
namespace engine {
namespace {

class BuildLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[engine_cache][TRT] " << msg << std::endl;
    }
  }
};

std::string lowerExtension(const std::string& path) {
  std::string ext = fs::path(path).extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext;
}

/// Compute capability of device 0, e.g. "sm89". Empty when unavailable.
std::string computeCapability() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return {};
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return {};
  return "sm" + std::to_string(prop.major) + std::to_string(prop.minor);
}

/// FNV-1a: short, dependency-free, and adequate for a cache filename.
std::string hashString(const std::string& input) {
  std::uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : input) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  std::ostringstream ss;
  ss << std::hex << h;
  return ss.str();
}

bool buildEngine(const fs::path& onnx, const fs::path& out,
                 const std::string& precision, std::string& error_out) {
  BuildLogger logger;

  auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    error_out = "failed to create TensorRT builder";
    return false;
  }

  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
  auto parser = std::unique_ptr<nvonnxparser::IParser>(
    nvonnxparser::createParser(*network, logger));

  if (!parser->parseFromFile(
        onnx.string().c_str(),
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    std::ostringstream ss;
    ss << "failed to parse " << onnx;
    for (int i = 0; i < parser->getNbErrors(); ++i) {
      ss << "; " << parser->getError(i)->desc();
    }
    error_out = ss.str();
    return false;
  }

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             4096ULL * 1024ULL * 1024ULL);
  if (precision == "fp16") {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
    builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    error_out = "TensorRT engine build failed";
    return false;
  }

  // Write to a temporary sibling then rename, so a crash mid-write cannot
  // leave a truncated engine that later looks like a valid cache hit.
  const fs::path tmp = out.string() + ".partial";
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) {
      error_out = "cannot write " + tmp.string();
      return false;
    }
    f.write(static_cast<const char*>(plan->data()),
            static_cast<std::streamsize>(plan->size()));
  }
  std::error_code ec;
  fs::rename(tmp, out, ec);
  if (ec) {
    error_out = "cannot move engine into place: " + ec.message();
    return false;
  }
  return true;
}

}  // namespace

bool isEnginePath(const std::string& path) {
  const std::string ext = lowerExtension(path);
  return ext == ".trt" || ext == ".engine";
}

std::string cacheKey(const std::string& onnx_path, const std::string& precision,
                     int dla_core) {
  std::error_code ec;
  const auto size = fs::file_size(onnx_path, ec);
  const std::uintmax_t size_val = ec ? 0 : size;

  std::uint64_t mtime = 0;
  const auto t = fs::last_write_time(onnx_path, ec);
  if (!ec) {
    mtime = static_cast<std::uint64_t>(t.time_since_epoch().count());
  }

  std::ostringstream ss;
  ss << fs::path(onnx_path).filename().string() << '|' << size_val << '|'
     << mtime << '|' << precision << '|' << dla_core << '|'
     << computeCapability() << '|' << NV_TENSORRT_MAJOR << '.'
     << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH;
  return hashString(ss.str());
}

fs::path resolveEngine(const std::string& model_path, const std::string& precision,
                       int dla_core, const std::string& cache_dir,
                       std::string& error_out) {
  error_out.clear();

  if (model_path.empty()) {
    error_out = "model_path is empty";
    return {};
  }
  if (isEnginePath(model_path)) {
    // Existence is checked by the library when it loads the engine, so that
    // the error surfaces with its context rather than being duplicated here.
    return fs::path(model_path);
  }
  if (lowerExtension(model_path) != ".onnx") {
    error_out = "model_path must be a .onnx, .trt or .engine file, got " + model_path;
    return {};
  }

  std::error_code ec;
  if (!fs::exists(model_path, ec)) {
    error_out = "model not found: " + model_path;
    return {};
  }

  fs::path dir = cache_dir.empty() ? fs::path(model_path).parent_path()
                                   : fs::path(cache_dir);
  if (dir.empty()) dir = fs::current_path();
  fs::create_directories(dir, ec);

  const std::string stem = fs::path(model_path).stem().string();
  const fs::path engine_path =
    dir / (stem + "." + cacheKey(model_path, precision, dla_core) + ".trt");

  if (fs::exists(engine_path, ec)) {
    std::cout << "[engine_cache] cache hit: " << engine_path << std::endl;
    return engine_path;
  }

  std::cout << "[engine_cache] building engine from " << model_path
            << " (this can take several minutes)" << std::endl;
  if (!buildEngine(model_path, engine_path, precision, error_out)) {
    return {};
  }
  std::cout << "[engine_cache] wrote " << engine_path << std::endl;
  return engine_path;
}

}  // namespace engine
}  // namespace ros2_yolos_trt
