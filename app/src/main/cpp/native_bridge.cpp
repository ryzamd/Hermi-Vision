#include <jni.h>
#include <android/log.h>
#include <string>

#include "engine/hermivision_pipeline.h"

#include <opencv2/core.hpp>

#define LOG_TAG "NativeBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ════════════════════════════════════════════════════════════════════════════
// Global pipeline instance
// Lives for the duration of a processing session (init → process → release)
// ════════════════════════════════════════════════════════════════════════════
static hermivision::HermiVisionPipeline* g_pipeline = nullptr;

extern "C" {

// ────────────────────────────────────────────────────────────────────────────
// initPipeline(delegateType, numThreads, modelPath) → boolean
// Called once before processing starts
// ────────────────────────────────────────────────────────────────────────────
JNIEXPORT jboolean JNICALL
Java_com_hermitech_hermivision_domain_inference_NativePipeline_nativeInitPipeline(
    JNIEnv* env, jobject /* this */,
    jint delegateType, jint numThreads, jstring modelPath
) {
    // Release previous pipeline if exists
    if (g_pipeline) {
        g_pipeline->release();
        delete g_pipeline;
        g_pipeline = nullptr;
    }

    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (!path) {
        LOGE("Failed to get model path string");
        return JNI_FALSE;
    }

    hermivision::PipelineConfig config;
    config.delegate      = static_cast<hermivision::DelegateType>(delegateType);
    config.numThreads    = numThreads;
    config.ballModelPath = std::string(path);

    env->ReleaseStringUTFChars(modelPath, path);

    g_pipeline = new hermivision::HermiVisionPipeline();
    bool ok = g_pipeline->init(config);

    if (!ok) {
        LOGE("Pipeline initialization FAILED");
        delete g_pipeline;
        g_pipeline = nullptr;
        return JNI_FALSE;
    }

    LOGI("Pipeline initialized from JNI");
    return JNI_TRUE;
}

// ────────────────────────────────────────────────────────────────────────────
// processFrame(matAddr, frameId, origWidth, origHeight) → float[4]
//
// Receives the native address of an OpenCV Mat (RGB).
// Returns: [ballVisible (0/1), ballX, ballY, ballScore]
//
// The Mat is created in Kotlin via OpenCV Java and its nativeObj address
// is passed here for zero-copy access.
// ────────────────────────────────────────────────────────────────────────────
JNIEXPORT jfloatArray JNICALL
Java_com_hermitech_hermivision_domain_inference_NativePipeline_nativeProcessFrame(
    JNIEnv* env, jobject /* this */,
    jlong matAddr, jint frameId, jint origWidth, jint origHeight
) {
    jfloatArray resultArray = env->NewFloatArray(4);

    if (!g_pipeline) {
        float empty[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        env->SetFloatArrayRegion(resultArray, 0, 4, empty);
        return resultArray;
    }

    // Get OpenCV Mat from native address — ZERO COPY
    // The Mat was created in Kotlin (e.g. from HardwareVideoDecoder)
    // and its nativeObj pointer is passed directly
    cv::Mat& rgbMat = *reinterpret_cast<cv::Mat*>(matAddr);

    auto result = g_pipeline->processFrame(
        rgbMat, frameId, origWidth, origHeight
    );

    float data[4] = {
        result.ballVisible ? 1.0f : 0.0f,
        result.ballX,
        result.ballY,
        result.ballScore
    };
    env->SetFloatArrayRegion(resultArray, 0, 4, data);

    return resultArray;
}

// ────────────────────────────────────────────────────────────────────────────
// releasePipeline() — free all native resources
// ────────────────────────────────────────────────────────────────────────────
JNIEXPORT void JNICALL
Java_com_hermitech_hermivision_domain_inference_NativePipeline_nativeReleasePipeline(
    JNIEnv* env, jobject /* this */
) {
    if (g_pipeline) {
        g_pipeline->release();
        delete g_pipeline;
        g_pipeline = nullptr;
        LOGI("Pipeline released from JNI");
    }
}

// ────────────────────────────────────────────────────────────────────────────
// getActiveDelegate() → String
// ────────────────────────────────────────────────────────────────────────────
JNIEXPORT jstring JNICALL
Java_com_hermitech_hermivision_domain_inference_NativePipeline_nativeGetActiveDelegate(
    JNIEnv* env, jobject /* this */
) {
    std::string delegate = g_pipeline ? g_pipeline->getActiveDelegate() : "none";
    return env->NewStringUTF(delegate.c_str());
}

} // extern "C"
