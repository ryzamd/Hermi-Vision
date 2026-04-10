#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace hermivision {

// ───── Detection result from any model ─────
struct Detection {
    float cx{}, cy{}, w{}, h{};   // center-format bbox (letterbox 640×640 space)
    float score{};
    int classId = 0;
};

// ───── Pipeline context — passed by reference through all models (zero-copy) ─────
struct FrameContext {
    cv::Mat rgbImage;        // Current frame (RGB, original resolution)
    int originalWidth = 0;
    int originalHeight = 0;
    int frameId = 0;

    // Ball Detection output
    bool ballVisible = false;
    float ballX = 0.0f;     // Original-frame coordinate
    float ballY = 0.0f;     // Original-frame coordinate
    float ballScore = 0.0f;

    // ── Future: Trajectory for Bounce AI ──
    // std::deque<cv::Point2f> trajectory;

    // ── Future: Court Detection output ──
    // std::vector<cv::Point2f> courtCorners;

    // ── Future: Bounce Detection output ──
    // bool bounceDetected = false;
    // cv::Point2f bouncePoint;
};

// ───── Hardware delegate preference ─────
// Ordinal values match Kotlin DelegateType.ordinal (NNAPI=0, GPU=1, CPU=2)
enum class DelegateType : int {
    NNAPI = 0,
    GPU   = 1,
    CPU   = 2
};

// ───── Pipeline configuration (received from Kotlin AIConfig via JNI) ─────
struct PipelineConfig {
    DelegateType delegate = DelegateType::CPU;
    int numThreads = 4;
    std::string ballModelPath;
    // Future: std::string courtModelPath;
    // Future: std::string bounceModelPath;
};

// ───── Result sent back to Kotlin (per-frame) ─────
struct FrameResult {
    int frameId;
    bool ballVisible;
    float ballX, ballY;
    float ballScore;
};

} // namespace hermivision
