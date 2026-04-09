package com.hermitech.hermivision.domain.inference

import android.util.Log

/**
 * Kotlin wrapper for the C++ AI Pipeline (JNI).
 *
 * This is a thin shell — ALL heavy computation happens in native C++:
 *   - Pre-processing (Letterbox, Normalize)
 *   - TFLite inference (with NNAPI/GPU/CPU delegate)
 *   - Post-processing (NMS, un-letterbox)
 *
 * Usage:
 *   val pipeline = NativePipeline()
 *   pipeline.init(delegateType = 2, numThreads = 4, modelPath = "/path/to/model.tflite")
 *   val result = pipeline.processFrame(mat.nativeObjAddr, frameId, origW, origH)
 *   pipeline.release()
 */
class NativePipeline {

    companion object {
        private const val TAG = "NativePipeline"

        init {
            System.loadLibrary("hermivision_native")
            Log.i(TAG, "hermivision_native library loaded")
        }
    }

    private var initialized = false

    /**
     * Initialize the C++ pipeline.
     *
     * @param delegateType DelegateType.ordinal (0=NNAPI, 1=GPU, 2=CPU)
     * @param numThreads   CPU thread count
     * @param modelPath    Absolute path to .tflite model file on disk
     * @return true if initialization succeeded
     */
    fun init(delegateType: Int, numThreads: Int, modelPath: String): Boolean {
        val ok = nativeInitPipeline(delegateType, numThreads, modelPath)
        initialized = ok
        if (ok) {
            Log.i(TAG, "Pipeline initialized: delegate=${getActiveDelegate()}")
        } else {
            Log.e(TAG, "Pipeline initialization FAILED")
        }
        return ok
    }

    /**
     * Process a single RGB frame through the AI pipeline.
     *
     * @param matAddr    Native address of OpenCV Mat (mat.nativeObjAddr)
     * @param frameId    Sequential frame number
     * @param origWidth  Original video/camera width
     * @param origHeight Original video/camera height
     * @return FloatArray[4]: [ballVisible (0/1), ballX, ballY, ballScore]
     */
    fun processFrame(matAddr: Long, frameId: Int, origWidth: Int, origHeight: Int): FloatArray {
        if (!initialized) return floatArrayOf(0f, 0f, 0f, 0f)
        return nativeProcessFrame(matAddr, frameId, origWidth, origHeight)
    }

    /**
     * Release all native resources.
     */
    fun release() {
        if (initialized) {
            nativeReleasePipeline()
            initialized = false
            Log.i(TAG, "Pipeline released")
        }
    }

    /**
     * Get the hardware delegate that was actually applied.
     * e.g. "NPU (NNAPI)", "GPU", "CPU (4 threads)"
     */
    fun getActiveDelegate(): String {
        return nativeGetActiveDelegate()
    }

    // ── JNI native methods ──
    private external fun nativeInitPipeline(delegateType: Int, numThreads: Int, modelPath: String): Boolean
    private external fun nativeProcessFrame(matAddr: Long, frameId: Int, origWidth: Int, origHeight: Int): FloatArray
    private external fun nativeReleasePipeline()
    private external fun nativeGetActiveDelegate(): String
}
