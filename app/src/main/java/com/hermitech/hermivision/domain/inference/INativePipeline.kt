package com.hermitech.hermivision.domain.inference

import java.nio.ByteBuffer

/**
 * SOLID (Dependency Inversion): Abstract interface for the native AI pipeline.
 *
 * Decouples [VideoProcessingWorker], [LiveViewModel], and future consumers
 * from the concrete [NativePipeline] JNI implementation, enabling:
 *   - Unit-testable ViewModels (mock implementation)
 *   - Future swap to a pure-Kotlin fallback pipeline if needed
 */
interface INativePipeline {

    /**
     * Initialize the C++ pipeline.
     * @return true if initialization succeeded
     */
    fun init(delegateType: Int, numThreads: Int, ballModelPath: String, courtModelPath: String = "", courtDelegateType: Int = 2): Boolean

    /** Submit YUV frame from decoder/camera to C++ FramePool (zero-copy). */
    fun submitYuvFrame(yBuffer: ByteBuffer, uvBuffer: ByteBuffer, width: Int, height: Int, yStride: Int, uvStride: Int, frameId: Int)

    /** Submit RGB frame via OpenCV Mat pointer to FramePool. */
    fun submitFrame(matAddr: Long, frameId: Int, origWidth: Int, origHeight: Int)

    /**
     * Process the latest frame through all AI models.
     * @return FloatArray[35]: [ballVisible, ballX, ballY, ballW, ballH, ballScore,
     *                          courtValid, kp1x, kp1y, ..., kp14x, kp14y]
     */
    fun processLatestFrame(): FloatArray

    /**
     * Extract 14 court keypoints from a result array.
     * @return FloatArray[28] (x,y pairs) or null if court not valid
     */
    fun getCourtKeypoints(result: FloatArray): FloatArray?

    /** Get the hardware delegate that was actually applied. */
    fun getActiveDelegate(): String

    /** Release all native resources. */
    fun release()
}