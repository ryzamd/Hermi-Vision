package com.hermitech.hermivision.domain.inference

import android.util.Log
import com.hermitech.hermivision.shared.domain.model.AIConfig
import com.hermitech.hermivision.shared.domain.model.BallFrame
import com.hermitech.hermivision.shared.domain.model.DelegateType
import com.hermitech.hermivision.shared.domain.model.YoloLetterbox
import com.hermitech.hermivision.shared.domain.usecase.ParseYoloBallFrameUseCase
import kotlinx.coroutines.channels.ReceiveChannel
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * YOLO-based ball detector using TFLite (end2end=False, universal format).
 *
 * Compatible with YOLOv8, YOLO11, YOLO26 exported with end2end=False.
 *
 * Model I/O:
 *   Input:  float32[1, 640, 640, 3]  — NHWC, normalized [0,1], RGB
 *   Output: float32[1, 5, 8400]      — raw detections (x_center, y_center, w, h, score) × 8400 anchors
 *
 * Pipeline:
 *   1. Preprocess: Letterbox resize 640×640, normalize [0,1], bulk copy to ByteBuffer
 *   2. TFLite inference (XNNPACK/GPU accelerated)
 *   3. Post-process: confidence filter → NMS → best detection → undo letterbox
 *   4. Return BallFrame (same interface as TrackNet)
 */
class YoloBallInferencer(private val sessionManager: TFLiteSessionManager) {

    companion object {
        private const val TAG = "YoloBallInferencer"
        // TODO: Switch model variant per device tier in the future
        private const val DEFAULT_MODEL = "YoloBall-nano-FP32.tflite"
        private const val INPUT_SIZE = 640
        private const val NUM_ANCHORS = 8400    // 80×80 + 40×40 + 20×20
        private const val NUM_VALUES = 5        // x_center, y_center, w, h, score (1 class)
    }

    private var interpreter: Interpreter? = null

    // Pre-allocated input buffer: 1 * 640 * 640 * 3 * 4 bytes
    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        .order(ByteOrder.nativeOrder())

    // Pre-allocated bulk float array for Mat→ByteBuffer transfer (1 JNI call instead of 409,600)
    private val bulkFloatArray = FloatArray(INPUT_SIZE * INPUT_SIZE * 3)

    // Pre-allocated output array: [1][5][8400]
    private val outputArray = Array(1) { Array(NUM_VALUES) { FloatArray(NUM_ANCHORS) } }

    // Reusable Mats to avoid GC pressure
    private val resizedMat = Mat()
    private val floatMat = Mat()
    private val tempResized = Mat()

    // Letterbox state (updated each frame)
    private var letterboxScale = 1.0f
    private var letterboxPadX = 0
    private var letterboxPadY = 0
    private val parseYoloBallFrame = ParseYoloBallFrameUseCase(numAnchors = NUM_ANCHORS)

    /**
     * Load the YOLO model with optimal configuration from DeviceProfiler.
     *
     * @param config AIConfig from DeviceProfiler benchmark.
     *               If null, uses defaults (FP16, NNAPI→GPU→CPU fallback).
     */
    fun loadModel(config: AIConfig? = null) {
        val modelName = config?.yoloModelName ?: DEFAULT_MODEL
        val delegate = config?.tfliteDelegate ?: DelegateType.NNAPI
        interpreter = sessionManager.loadInterpreter(modelName, delegate)
        Log.i(TAG, "YoloBall loaded: $modelName on ${sessionManager.activeHardware}")
    }

    fun releaseModel() {
        interpreter?.close()
        interpreter = null
        resizedMat.release()
        floatMat.release()
        tempResized.release()
        Log.i(TAG, "YoloBall model released")
    }

    /**
     * Inference from decoded frame channel.
     * Compatible interface with TrackNetInferencer.inferFromChannel().
     */
    suspend fun inferFromChannel(inputChannel: ReceiveChannel<Mat>, origWidth: Int, origHeight: Int, onProgress: (Int) -> Unit = {}): List<BallFrame> {
        val interp = interpreter
            ?: throw IllegalStateException("Model not loaded. Call loadModel() first.")

        val results = ArrayList<BallFrame>()
        var frameId = 0

        for (mat in inputChannel) {
            val ballFrame = detectSingleFrame(interp, mat, frameId, origWidth, origHeight)
            results.add(ballFrame)
            mat.release()

            frameId++
            onProgress(frameId)
        }

        return results
    }

    /**
     * Detect ball in a single frame.
     */
    private fun detectSingleFrame(interp: Interpreter, mat: Mat, frameId: Int, origWidth: Int, origHeight: Int): BallFrame {
        // 1. Preprocess: Letterbox resize + normalize + bulk copy
        preprocessFrame(mat)

        // 2. Run TFLite inference
        interp.run(inputBuffer, outputArray)

        // 3. Post-process: confidence filter → NMS → best detection
        return parseOutput(frameId)
    }

    /**
     * Optimized preprocessing pipeline:
     *  1. Letterbox resize (preserve aspect ratio, pad with black)
     *  2. Convert to float32 and normalize [0, 1]
     *  3. Bulk copy Mat → FloatArray → ByteBuffer (1 JNI call)
     *
     * NOTE: HardwareVideoDecoder already outputs RGB (via YUV2RGB_NV21),
     * so NO BGR→RGB conversion is needed here.
     */
    private fun preprocessFrame(mat: Mat) {
        // Step 1: Letterbox resize — keep aspect ratio, pad black
        letterboxResize(mat)

        // Step 2: Convert to float [0, 1]
        resizedMat.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0)

        // Step 3: Bulk copy — 1 single JNI call instead of 409,600 pixel-by-pixel calls
        floatMat.get(0, 0, bulkFloatArray)

        inputBuffer.rewind()
        inputBuffer.asFloatBuffer().put(bulkFloatArray)
        inputBuffer.rewind()
    }

    /**
     * Letterbox resize: scales image to fit INPUT_SIZE while preserving aspect ratio.
     * Remaining space is filled with black (0,0,0) padding.
     * Stores scale/offset for coordinate mapping in parseOutput().
     */
    private fun letterboxResize(src: Mat) {
        val srcW = src.cols()
        val srcH = src.rows()
        val scale = minOf(INPUT_SIZE.toFloat() / srcW, INPUT_SIZE.toFloat() / srcH)
        val newW = (srcW * scale).toInt()
        val newH = (srcH * scale).toInt()
        val padX = (INPUT_SIZE - newW) / 2
        val padY = (INPUT_SIZE - newH) / 2

        letterboxScale = scale
        letterboxPadX = padX
        letterboxPadY = padY

        // Resize preserving aspect ratio
        Imgproc.resize(src, tempResized, Size(newW.toDouble(), newH.toDouble()))

        // Create black canvas and paste resized image in center
        resizedMat.create(INPUT_SIZE, INPUT_SIZE, src.type())
        resizedMat.setTo(Scalar(0.0, 0.0, 0.0))
        val roi = resizedMat.submat(Rect(padX, padY, newW, newH))
        tempResized.copyTo(roi)
    }

    // ==================== POST-PROCESSING (NMS) ====================

    /**
     * Parse raw YOLO output [1, 5, 8400] → best ball detection.
     *
     * Output layout (per anchor i, for single-class detection):
     *   outputArray[0][0][i] = x_center  (in letterbox pixel space, 0..640)
     *   outputArray[0][1][i] = y_center
     *   outputArray[0][2][i] = width
     *   outputArray[0][3][i] = height
     *   outputArray[0][4][i] = class_score (ball confidence)
     *
     * Steps:
     *   1. Filter by confidence threshold
     *   2. Apply NMS to remove overlapping boxes
     *   3. Pick highest confidence detection
     *   4. Undo letterbox to get original frame coordinates
     */
    private fun parseOutput(frameId: Int): BallFrame = parseYoloBallFrame(
        frameId = frameId,
        output = outputArray[0],
        letterbox = YoloLetterbox(
            scale = letterboxScale,
            padX = letterboxPadX,
            padY = letterboxPadY,
        ),
    )
}
