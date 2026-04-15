package com.hermitech.hermivision.data.inference

import com.hermitech.hermivision.domain.inference.INativePipeline
import com.hermitech.hermivision.domain.inference.DelegateType
import com.hermitech.hermivision.domain.inference.AIConfig
import com.hermitech.hermivision.domain.inference.DeviceTier

import android.util.Log
import java.nio.ByteBuffer

/**
 * Kotlin wrapper for the C++ AI Pipeline (JNI).
 *
 * This is a thin shell — ALL heavy computation happens in native C++:
 *   - YUV→RGB conversion (NEON SIMD)
 *   - Pre-processing (Letterbox, Normalize)
 *   - TFLite inference (with NNAPI/GPU/CPU delegate)
 *   - Post-processing (NMS, un-letterbox)
 *
 * Usage (FramePool flow):
 *   val pipeline = NativePipeline()
 *   pipeline.init(delegateType = 2, numThreads = 4, ballModelPath = "/path/to/yolo.tflite")
 *   // Decoder submits frames:
 *   pipeline.submitYuvFrame(yBuffer, uvBuffer, width, height, yStride, uvStride, frameId)
 *   // Consumer processes:
 *   val result = pipeline.processLatestFrame()
 *   pipeline.release()
 */
class NativePipeline : INativePipeline {

    companion object {
        private const val TAG = "NativePipeline"

        // Result array layout: float[35]
        // Ball results (indices 0-5)
        const val IDX_BALL_VISIBLE = 0
        const val IDX_BALL_X = 1
        const val IDX_BALL_Y = 2
        const val IDX_BALL_W = 3
        const val IDX_BALL_H = 4
        const val IDX_BALL_SCORE = 5
        // Court results (indices 6-34)
        const val IDX_COURT_VALID = 6
        const val IDX_COURT_KP_START = 7  // 28 floats: kp1x, kp1y, ..., kp14x, kp14y

        const val RESULT_SIZE = 35  // 6 (ball) + 1 (courtValid) + 28 (keypoints)

        init {
            System.loadLibrary("hermivision_native")
            Log.i(TAG, "hermivision_native library loaded")
        }
    }

    private var initialized = false

    /**
     * Initialize the C++ pipeline.
     *
     * @param delegateType      DelegateType.ordinal (0=NNAPI, 1=GPU, 2=CPU) for ball model
     * @param numThreads        CPU thread count
     * @param ballModelPath     Absolute path to ball detection .tflite model
     * @param courtModelPath    Absolute path to court detection .tflite model (empty = disabled)
     * @param courtDelegateType DelegateType.ordinal for court model (default CPU=2)
     * @return true if initialization succeeded
     */
    override fun init(
        delegateType: Int,
        numThreads: Int,
        ballModelPath: String,
        courtModelPath: String,
        courtDelegateType: Int
    ): Boolean {
        val ok = nativeInitPipeline(delegateType, numThreads, ballModelPath, courtModelPath, courtDelegateType)
        initialized = ok
        if (ok) {
            Log.i(TAG, "Pipeline initialized: delegate=${getActiveDelegate()}")
        } else {
            Log.e(TAG, "Pipeline initialization FAILED")
        }
        return ok
    }

    // ── Frame Submission (Producer API) ──

    /**
     * Submit YUV frame from decoder/camera directly to C++ FramePool.
     * YUV→RGB conversion happens in C++ using NEON SIMD — zero Kotlin allocation.
     *
     * @param yBuffer   Direct ByteBuffer for Y plane
     * @param uvBuffer  Direct ByteBuffer for UV/VU plane (NV21 interleaved)
     * @param width     Frame width in pixels
     * @param height    Frame height in pixels
     * @param yStride   Row stride of Y plane
     * @param uvStride  Row stride of UV plane
     * @param frameId   Sequential frame number
     */
    override fun submitYuvFrame(yBuffer: ByteBuffer, uvBuffer: ByteBuffer, width: Int, height: Int, yStride: Int, uvStride: Int, frameId: Int) {
        if (!initialized) return
        nativeSubmitYuvFrame(yBuffer, uvBuffer, width, height, yStride, uvStride, frameId)
    }

    /**
     * Submit RGB frame via OpenCV Mat pointer to FramePool.
     *
     * @param matAddr    Native address of OpenCV Mat (mat.nativeObjAddr)
     * @param frameId    Sequential frame number
     * @param origWidth  Original frame width
     * @param origHeight Original frame height
     */
    override fun submitFrame(matAddr: Long, frameId: Int, origWidth: Int, origHeight: Int) {
        if (!initialized) return
        nativeSubmitFrame(matAddr, frameId, origWidth, origHeight)
    }

    // ── Processing (Consumer API) ──

    /**
     * Process the latest frame in the FramePool through all AI models.
     * @return FloatArray[33]: [ballVisible, ballX, ballY, ballScore,
     *                          courtValid, kp1x, kp1y, ..., kp14x, kp14y]
     */
    override fun processLatestFrame(): FloatArray {
        if (!initialized) return FloatArray(RESULT_SIZE)
        return nativeProcessLatestFrame()
    }

    /**
     * [LEGACY] Process a single RGB frame — submits to pool then processes.
     * Prefer submitFrame() + processLatestFrame() for new code.
     */
    fun processFrame(matAddr: Long, frameId: Int, origWidth: Int, origHeight: Int): FloatArray {
        if (!initialized) return FloatArray(RESULT_SIZE)
        return nativeProcessFrame(matAddr, frameId, origWidth, origHeight)
    }

    /**
     * Extract 14 court keypoints (x, y pairs in original pixel coords) from a result array.
     * @return FloatArray[28] or null if court detection is not valid
     */
    override fun getCourtKeypoints(result: FloatArray): FloatArray? {
        if (result.size < RESULT_SIZE || result[IDX_COURT_VALID] <= 0.5f) return null
        return result.copyOfRange(IDX_COURT_KP_START, IDX_COURT_KP_START + 28)
    }

    /**
     * Release all native resources.
     */
    override fun release() {
        if (initialized) {
            nativeReleasePipeline()
            initialized = false
            Log.i(TAG, "Pipeline released")
        }
    }

    /**
     * Get the hardware delegate that was actually applied.
     */
    override fun getActiveDelegate(): String {
        return nativeGetActiveDelegate()
    }

    // ── JNI native methods ──
    private external fun nativeSubmitYuvFrame(yBuffer: ByteBuffer, uvBuffer: ByteBuffer, width: Int, height: Int, yStride: Int, uvStride: Int, frameId: Int)
    private external fun nativeInitPipeline(delegateType: Int, numThreads: Int, ballModelPath: String, courtModelPath: String, courtDelegateType: Int): Boolean
    private external fun nativeSubmitFrame(matAddr: Long, frameId: Int, origWidth: Int, origHeight: Int)
    private external fun nativeProcessLatestFrame(): FloatArray
    private external fun nativeProcessFrame(matAddr: Long, frameId: Int, origWidth: Int, origHeight: Int): FloatArray
    private external fun nativeReleasePipeline()
    private external fun nativeGetActiveDelegate(): String
}