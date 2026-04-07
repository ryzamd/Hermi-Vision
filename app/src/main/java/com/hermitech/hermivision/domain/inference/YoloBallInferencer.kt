package com.hermitech.hermivision.domain.inference

import android.util.Log
import com.hermitech.hermivision.data.model.BallFrame
import kotlinx.coroutines.channels.ReceiveChannel
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * YOLO26-based ball detector using TFLite.
 * Replaces TrackNetV3 — single frame input, bounding box output.
 *
 * YOLO26 advantages: NMS-Free, STAL (small-target-aware), 43% faster CPU.
 *
 * Model I/O:
 *   Input:  float32[1, 640, 640, 3]  — NHWC, normalized [0,1], RGB
 *   Output: float32[1, 300, 6]       — 300 detections, (x1,y1,x2,y2,conf,class)
 *
 * Pipeline:
 *   1. Preprocess: BGR→RGB, resize 640×640, normalize [0,1], bulk copy to ByteBuffer
 *   2. TFLite inference (XNNPACK/GPU accelerated)
 *   3. Post-process: filter by confidence → best detection
 *   4. Return BallFrame (same interface as TrackNet)
 */
class YoloBallInferencer(private val sessionManager: TFLiteSessionManager) {

    companion object {
        private const val TAG = "YoloBallInferencer"
        private const val MODEL_NAME = "YoloBall.tflite"
        private const val INPUT_SIZE = 640
        private const val CONF_THRESHOLD = 0.25f
        private const val NUM_DETECTIONS = 300
        private const val NUM_VALUES = 6  // x1, y1, x2, y2, conf, class_id
    }

    private var interpreter: Interpreter? = null

    // Pre-allocated input buffer: 1 * 640 * 640 * 3 * 4 bytes
    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        .order(ByteOrder.nativeOrder())

    // Pre-allocated bulk float array for Mat→ByteBuffer transfer (1 JNI call instead of 409,600)
    private val bulkFloatArray = FloatArray(INPUT_SIZE * INPUT_SIZE * 3)

    // Pre-allocated output array: [1][300][6]
    private val outputArray = Array(1) { Array(NUM_DETECTIONS) { FloatArray(NUM_VALUES) } }

    // Reusable Mats to avoid GC pressure
    private val resizedMat = Mat()
    private val rgbMat = Mat()
    private val floatMat = Mat()

    fun loadModel() {
        interpreter = sessionManager.loadInterpreter(MODEL_NAME)
        Log.i(TAG, "YoloBall model loaded (FP16, ${INPUT_SIZE}×${INPUT_SIZE})")
    }

    fun releaseModel() {
        interpreter?.close()
        interpreter = null
        resizedMat.release()
        rgbMat.release()
        floatMat.release()
        Log.i(TAG, "YoloBall model released")
    }

    /**
     * Inference from decoded frame channel.
     * Compatible interface with TrackNetInferencer.inferFromChannel().
     */
    suspend fun inferFromChannel(
        inputChannel: ReceiveChannel<Mat>,
        origWidth: Int,
        origHeight: Int,
        onProgress: (Int) -> Unit = {}
    ): List<BallFrame> {
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
    private fun detectSingleFrame(
        interp: Interpreter,
        mat: Mat,
        frameId: Int,
        origWidth: Int,
        origHeight: Int
    ): BallFrame {
        // 1. Preprocess: BGR→RGB + resize + normalize + bulk copy
        preprocessFrame(mat)

        // 2. Run TFLite inference
        interp.run(inputBuffer, outputArray)

        // 3. Post-process: find best detection
        return parseOutput(frameId, origWidth, origHeight)
    }

    /**
     * Optimized preprocessing pipeline:
     *  1. BGR → RGB (OpenCV decodes as BGR, YOLO expects RGB)
     *  2. Resize to INPUT_SIZE × INPUT_SIZE
     *  3. Convert to float32 and normalize [0, 1]
     *  4. Bulk copy Mat → FloatArray → ByteBuffer (1 JNI call)
     */
    private fun preprocessFrame(mat: Mat) {
        // Step 1: BGR → RGB conversion (CRITICAL: OpenCV=BGR, YOLO=RGB)
        Imgproc.cvtColor(mat, rgbMat, Imgproc.COLOR_BGR2RGB)

        // Step 2: Resize to INPUT_SIZE × INPUT_SIZE
        Imgproc.resize(rgbMat, resizedMat, Size(INPUT_SIZE.toDouble(), INPUT_SIZE.toDouble()))

        // Step 3: Convert to float [0, 1]
        resizedMat.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0)

        // Step 4: Bulk copy — 1 single JNI call instead of 409,600 pixel-by-pixel calls
        floatMat.get(0, 0, bulkFloatArray)

        inputBuffer.rewind()
        inputBuffer.asFloatBuffer().put(bulkFloatArray)
        inputBuffer.rewind()
    }

    /**
     * Parse TFLite output [1, 300, 6] → best ball detection.
     *
     * Output format per detection: [x1, y1, x2, y2, confidence, class_id]
     * Coordinates are normalized [0, 1].
     */
    private fun parseOutput(
        frameId: Int,
        origWidth: Int,
        origHeight: Int
    ): BallFrame {
        val detections = outputArray[0]  // [300, 6]

        // Find detection with highest confidence
        var bestConf = 0f
        var bestIdx = -1

        for (i in 0 until NUM_DETECTIONS) {
            val conf = detections[i][4]
            if (conf > CONF_THRESHOLD && conf > bestConf) {
                bestConf = conf
                bestIdx = i
            }
        }

        if (bestIdx < 0) {
            return BallFrame(frameId, isVisible = false, x = null, y = null)
        }

        // Get center from bounding box (x1, y1, x2, y2) — normalized [0,1]
        val x1 = detections[bestIdx][0]
        val y1 = detections[bestIdx][1]
        val x2 = detections[bestIdx][2]
        val y2 = detections[bestIdx][3]

        val cx = (x1 + x2) / 2f
        val cy = (y1 + y2) / 2f

        // Coordinates are normalized [0,1] — scale to original frame size directly
        val x = cx * origWidth
        val y = cy * origHeight

        return BallFrame(frameId, isVisible = true, x = x, y = y)
    }
}
