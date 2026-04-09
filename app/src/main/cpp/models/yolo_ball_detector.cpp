#include "yolo_ball_detector.h"

#include <android/log.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstring>

#define LOG_TAG "YoloBallDetector"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace hermivision {

// ════════════════════════════════════════════════════════════════════════════
// Model Lifecycle
// ════════════════════════════════════════════════════════════════════════════

bool YoloBallDetector::loadModel(
    const std::string& modelPath,
    DelegateType delegate,
    int numThreads
) {
    // 1. Load model from file
    model_ = TfLiteModelCreateFromFile(modelPath.c_str());
    if (!model_) {
        LOGE("Failed to load model: %s", modelPath.c_str());
        return false;
    }

    // 2. Create interpreter with delegate auto-fallback
    TfLiteInterpreter* interp = delegateManager_.createInterpreter(model_, delegate, numThreads);
    if (!interp) {
        LOGE("Failed to create interpreter for: %s", modelPath.c_str());
        TfLiteModelDelete(model_);
        model_ = nullptr;
        return false;
    }

    // 3. Validate tensor shapes
    const TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(interp, 0);
    if (TfLiteTensorNumDims(inputTensor) != 4) {
        LOGE("Unexpected input tensor dims: %d (expected 4)", TfLiteTensorNumDims(inputTensor));
        release();
        return false;
    }

    // 4. Pre-allocate reusable cv::Mat buffers (ONE-TIME allocation)
    resizedMat_ = cv::Mat(INPUT_SIZE, INPUT_SIZE, CV_8UC3);
    floatMat_   = cv::Mat(INPUT_SIZE, INPUT_SIZE, CV_32FC3);

    LOGI("YoloBall loaded: %s on %s", modelPath.c_str(),
         delegateManager_.getActiveDelegateName().c_str());
    LOGI("  Input:  [1, %d, %d, 3] float32",
         TfLiteTensorDim(inputTensor, 1), TfLiteTensorDim(inputTensor, 2));

    const TfLiteTensor* outputTensor = TfLiteInterpreterGetOutputTensor(interp, 0);
    LOGI("  Output: [1, %d, %d] float32",
         TfLiteTensorDim(outputTensor, 1), TfLiteTensorDim(outputTensor, 2));

    return true;
}

void YoloBallDetector::release() {
    delegateManager_.release();   // Releases interpreter + delegate

    if (model_) {
        TfLiteModelDelete(model_);
        model_ = nullptr;
    }

    resizedMat_.release();
    floatMat_.release();
    tempResized_.release();

    LOGI("YoloBall model released");
}

std::string YoloBallDetector::getActiveDelegate() const {
    return delegateManager_.getActiveDelegateName();
}

// ════════════════════════════════════════════════════════════════════════════
// Main Process — called once per frame
// ════════════════════════════════════════════════════════════════════════════

void YoloBallDetector::process(FrameContext& ctx) {
    TfLiteInterpreter* interp = delegateManager_.getInterpreter();
    if (!interp) {
        ctx.ballVisible = false;
        return;
    }

    // 1. Pre-process: Letterbox + Normalize → copy to TFLite input tensor
    preprocess(ctx.rgbImage);

    // 2. Invoke inference
    if (TfLiteInterpreterInvoke(interp) != kTfLiteOk) {
        LOGW("TFLite invoke failed on frame %d", ctx.frameId);
        ctx.ballVisible = false;
        return;
    }

    // 3. Post-process: Parse output → NMS → un-letterbox → write to ctx
    postprocess(ctx);
}

// ════════════════════════════════════════════════════════════════════════════
// Pre-processing: Letterbox + Normalize
// ════════════════════════════════════════════════════════════════════════════

void YoloBallDetector::preprocess(const cv::Mat& rgbFrame) {
    // ── Step 1: Letterbox resize (preserve aspect ratio, pad black) ──
    const int srcW = rgbFrame.cols;
    const int srcH = rgbFrame.rows;

    letterboxScale_ = std::min(
        static_cast<float>(INPUT_SIZE) / static_cast<float>(srcW),
        static_cast<float>(INPUT_SIZE) / static_cast<float>(srcH)
    );

    const int newW = static_cast<int>(srcW * letterboxScale_);
    const int newH = static_cast<int>(srcH * letterboxScale_);
    letterboxPadX_ = (INPUT_SIZE - newW) / 2;
    letterboxPadY_ = (INPUT_SIZE - newH) / 2;

    // Resize to fit within 640×640 (preserving aspect ratio)
    cv::resize(rgbFrame, tempResized_, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    // Create black 640×640 canvas and paste resized image in center
    resizedMat_.setTo(cv::Scalar(0, 0, 0));
    cv::Rect roi(letterboxPadX_, letterboxPadY_, newW, newH);
    tempResized_.copyTo(resizedMat_(roi));

    // ── Step 2: Normalize [0, 255] → [0.0, 1.0] float32 ──
    resizedMat_.convertTo(floatMat_, CV_32FC3, 1.0 / 255.0);

    // ── Step 3: Copy to TFLite input tensor (memcpy to pre-allocated buffer) ──
    TfLiteInterpreter* interp = delegateManager_.getInterpreter();
    TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(interp, 0);

    // Direct memcpy — floatMat_ data layout matches NHWC [1, 640, 640, 3]
    TfLiteTensorCopyFromBuffer(
        inputTensor,
        floatMat_.data,
        INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float)
    );
}

// ════════════════════════════════════════════════════════════════════════════
// Post-processing: Parse [1, 5, 8400] → NMS → un-letterbox
// ════════════════════════════════════════════════════════════════════════════

void YoloBallDetector::postprocess(FrameContext& ctx) {
    TfLiteInterpreter* interp = delegateManager_.getInterpreter();
    const TfLiteTensor* outputTensor = TfLiteInterpreterGetOutputTensor(interp, 0);

    // Get raw output pointer — layout: [1, 5, 8400] row-major
    // Row 0 (offset 0*8400): x_center for all 8400 anchors
    // Row 1 (offset 1*8400): y_center
    // Row 2 (offset 2*8400): width
    // Row 3 (offset 3*8400): height
    // Row 4 (offset 4*8400): ball_score
    const float* output = reinterpret_cast<const float*>(TfLiteTensorData(outputTensor));

    // ── Step 1: Filter by confidence threshold ──
    std::vector<Detection> candidates;
    candidates.reserve(MAX_CANDIDATES);

    for (int i = 0; i < NUM_ANCHORS; ++i) {
        const float score = output[4 * NUM_ANCHORS + i];   // Row 4

        if (score > CONF_THRESHOLD) {
            Detection det;
            det.cx    = output[0 * NUM_ANCHORS + i];       // Row 0
            det.cy    = output[1 * NUM_ANCHORS + i];       // Row 1
            det.w     = output[2 * NUM_ANCHORS + i];       // Row 2
            det.h     = output[3 * NUM_ANCHORS + i];       // Row 3
            det.score = score;
            det.classId = 0;
            candidates.push_back(det);
        }
    }

    if (candidates.empty()) {
        ctx.ballVisible = false;
        return;
    }

    // ── Step 2: Sort by confidence descending ──
    std::sort(candidates.begin(), candidates.end(),
        [](const Detection& a, const Detection& b) { return a.score > b.score; });

    // ── Step 3: Apply NMS ──
    std::vector<Detection> kept;
    nms(candidates, IOU_THRESHOLD, kept);

    if (kept.empty()) {
        ctx.ballVisible = false;
        return;
    }

    // ── Step 4: Best detection → un-letterbox to original coordinates ──
    const Detection& best = kept[0];

    // Remove letterbox padding, then undo scale
    const float rawX = best.cx - static_cast<float>(letterboxPadX_);
    const float rawY = best.cy - static_cast<float>(letterboxPadY_);

    ctx.ballVisible = true;
    ctx.ballX       = rawX / letterboxScale_;
    ctx.ballY       = rawY / letterboxScale_;
    ctx.ballScore   = best.score;
}

// ════════════════════════════════════════════════════════════════════════════
// NMS (Non-Maximum Suppression)
// ════════════════════════════════════════════════════════════════════════════

float YoloBallDetector::computeIoU(const Detection& a, const Detection& b) {
    // Convert center-format → corner-format
    const float ax1 = a.cx - a.w * 0.5f;
    const float ay1 = a.cy - a.h * 0.5f;
    const float ax2 = a.cx + a.w * 0.5f;
    const float ay2 = a.cy + a.h * 0.5f;

    const float bx1 = b.cx - b.w * 0.5f;
    const float by1 = b.cy - b.h * 0.5f;
    const float bx2 = b.cx + b.w * 0.5f;
    const float by2 = b.cy + b.h * 0.5f;

    // Intersection
    const float interW = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1));
    const float interH = std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
    const float interArea = interW * interH;

    // Union
    const float unionArea = a.w * a.h + b.w * b.h - interArea;
    return unionArea > 0.0f ? interArea / unionArea : 0.0f;
}

void YoloBallDetector::nms(
    std::vector<Detection>& candidates,
    float iouThreshold,
    std::vector<Detection>& kept
) {
    kept.clear();
    std::vector<bool> suppressed(candidates.size(), false);

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;

        kept.push_back(candidates[i]);

        // Suppress all lower-confidence detections that overlap
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (!suppressed[j] && computeIoU(candidates[i], candidates[j]) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }
}

} // namespace hermivision
