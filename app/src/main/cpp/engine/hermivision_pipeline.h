#pragma once

#include "types.h"
#include "tracker.h"
#include "../models/yolo_ball_detector.h"
#include <memory>
#include <string>

namespace hermivision {

/**
 * HermiVision Pipeline — orchestrates all AI models for a single frame.
 *
 * Current (Phase 1): Ball Detection + Kalman Tracker
 * Future:            Court Detection, Bounce Detection
 *
 * Lifecycle:
 *   1. init(config)      — called once from JNI initPipeline()
 *   2. processFrame(...)  — called per-frame from JNI processFrame()
 *   3. release()         — called from JNI releasePipeline()
 */
class HermiVisionPipeline {
public:
    HermiVisionPipeline() = default;
    ~HermiVisionPipeline() { release(); }

    /// Initialize pipeline with config from Kotlin AIConfig
    bool init(const PipelineConfig& config);

    /// Process a single RGB frame → returns FrameResult
    FrameResult processFrame(const cv::Mat& rgbFrame, int frameId, int origW, int origH);

    /// Release all model resources
    void release();

    /// Query active hardware delegate
    std::string getActiveDelegate() const;

private:
    PipelineConfig config_;

    // ── AI Models ──
    std::unique_ptr<YoloBallDetector> ballDetector_;

    // ── Ball Tracker (Kalman Filter — pure math, not AI) ──
    BallTracker tracker_;

    // ── Future models ──
    // std::unique_ptr<CourtDetector>  courtDetector_;
    // std::unique_ptr<BounceDetector> bounceDetector_;

    bool initialized_ = false;
};

} // namespace hermivision
