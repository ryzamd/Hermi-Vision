#include "hermivision_pipeline.h"
#include <android/log.h>

#define LOG_TAG "HermiPipeline"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace hermivision {

bool HermiVisionPipeline::init(const PipelineConfig& config) {
    config_ = config;

    // ── Initialize Ball Detector ──
    ballDetector_ = std::make_unique<YoloBallDetector>();
    if (!ballDetector_->loadModel(config.ballModelPath, config.delegate, config.numThreads)) {
        LOGE("Failed to load ball detection model: %s", config.ballModelPath.c_str());
        ballDetector_.reset();
        return false;
    }

    // ── Initialize Ball Tracker (Kalman Filter — pure math) ──
    tracker_.reset();

    initialized_ = true;
    LOGI("═══════════════════════════════════════════════");
    LOGI("Pipeline initialized successfully");
    LOGI("  Ball model: %s", config.ballModelPath.c_str());
    LOGI("  Delegate:   %s", ballDetector_->getActiveDelegate().c_str());
    LOGI("  Threads:    %d", config.numThreads);
    LOGI("  Tracker:    Kalman Filter (max_miss=%d)", 5);
    LOGI("═══════════════════════════════════════════════");

    return true;
}

FrameResult HermiVisionPipeline::processFrame(
    const cv::Mat& rgbFrame,
    int frameId,
    int origW,
    int origH
) {
    FrameResult result{};
    result.frameId = frameId;

    if (!initialized_ || !ballDetector_) {
        return result;
    }

    // ── Build FrameContext (zero-copy: cv::Mat header only, no pixel copy) ──
    FrameContext ctx;
    ctx.rgbImage       = rgbFrame;   // Shallow copy — shares pixel data
    ctx.originalWidth  = origW;
    ctx.originalHeight = origH;
    ctx.frameId        = frameId;

    // ── Stage 1: Ball Detection (YOLO — runs every frame) ──
    ballDetector_->process(ctx);

    // ── Stage 2: Kalman Tracker (smooth trajectory, fill occlusion gaps) ──
    if (ctx.ballVisible) {
        // YOLO found the ball → update tracker with real coordinates
        tracker_.update(ctx.ballX, ctx.ballY);
        // ctx.ballX/Y already set correctly by YOLO
    } else {
        // YOLO missed the ball → let tracker predict position
        if (tracker_.isInitialized() && !tracker_.isLost()) {
            cv::Point2f predicted = tracker_.predict();
            if (!tracker_.isLost()) {
                // Tracker still confident — use predicted position
                ctx.ballVisible = true;
                ctx.ballX = predicted.x;
                ctx.ballY = predicted.y;
                ctx.ballScore = 0.1f;  // Low score indicates prediction, not detection
            }
            // If tracker just became lost after this predict(), leave ballVisible = false
        }
    }

    // ── Future Stage 3: Court Detection (every N frames) ──
    // if (frameId % 30 == 0) { courtDetector_->process(ctx); }

    // ── Future Stage 4: Bounce Detection (when trajectory ready) ──
    // if (ctx.trajectory.size() >= 30) { bounceDetector_->process(ctx); }

    // ── Pack results for Kotlin ──
    result.ballVisible = ctx.ballVisible;
    result.ballX       = ctx.ballX;
    result.ballY       = ctx.ballY;
    result.ballScore   = ctx.ballScore;

    return result;
}

void HermiVisionPipeline::release() {
    if (ballDetector_) {
        ballDetector_->release();
        ballDetector_.reset();
    }
    tracker_.reset();
    // Future: courtDetector_->release(); bounceDetector_->release();

    initialized_ = false;
    LOGI("Pipeline released");
}

std::string HermiVisionPipeline::getActiveDelegate() const {
    if (ballDetector_) {
        return ballDetector_->getActiveDelegate();
    }
    return "none";
}

} // namespace hermivision
