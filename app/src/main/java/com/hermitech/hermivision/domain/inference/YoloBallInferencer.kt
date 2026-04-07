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
 *   Input:  float32[1, 640, 640, 3]  — NHWC, normalized [0,1]
 *   Output: float32[1, 300, 6]       — 300 detections, (x1,y1,x2,y2,conf,class)
 *
 * Pipeline:
 *   1. Preprocess: resize 640×640, normalize [0,1], NHWC layout
 *   2. TFLite inference
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

    // Pre-allocated output array: [1][300][6]
    private val outputArray = Array(1) { Array(NUM_DETECTIONS) { FloatArray(NUM_VALUES) } }

    fun loadModel() {
        interpreter = sessionManager.loadInterpreter(MODEL_NAME)
        Log.i(TAG, "YoloBall model loaded")
    }

    fun releaseModel() {
        interpreter?.close()
        interpreter = null
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
        // 1. Preprocess: resize + normalize + NHWC
        preprocessFrame(mat)

        // 2. Run TFLite inference
        interp.run(inputBuffer, outputArray)

        // 3. Post-process: find best detection
        return parseOutput(frameId, origWidth, origHeight)
    }

    /**
     * Preprocess: resize to 640×640, normalize [0,1], write to NHWC ByteBuffer.
     */
    private fun preprocessFrame(mat: Mat) {
        // Resize to INPUT_SIZE × INPUT_SIZE
        val resized = Mat()
        Imgproc.resize(mat, resized, Size(INPUT_SIZE.toDouble(), INPUT_SIZE.toDouble()))

        // Convert to float [0, 1]
        val floatMat = Mat()
        resized.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0)
        resized.release()

        // Write to ByteBuffer in NHWC order (row by row, pixel by pixel, RGB)
        inputBuffer.rewind()
        val pixelBuffer = FloatArray(3)
        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                floatMat.get(y, x, pixelBuffer) // Returns [R, G, B] for CV_32FC3
                inputBuffer.putFloat(pixelBuffer[0]) // R
                inputBuffer.putFloat(pixelBuffer[1]) // G
                inputBuffer.putFloat(pixelBuffer[2]) // B
            }
        }
        floatMat.release()
        inputBuffer.rewind()
    }

    /**
     * Parse TFLite output [1, 300, 6] → best ball detection.
     *
     * Output format per detection: [x1, y1, x2, y2, confidence, class_id]
     * Coordinates are in pixel space relative to INPUT_SIZE (640×640).
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

        // Get center from bounding box (x1, y1, x2, y2)
        val x1 = detections[bestIdx][0]
        val y1 = detections[bestIdx][1]
        val x2 = detections[bestIdx][2]
        val y2 = detections[bestIdx][3]

        val cx = (x1 + x2) / 2f
        val cy = (y1 + y2) / 2f

        // Scale back to original frame size
        val scaleX = origWidth.toFloat() / INPUT_SIZE
        val scaleY = origHeight.toFloat() / INPUT_SIZE

        val x = cx * scaleX
        val y = cy * scaleY

        return BallFrame(frameId, isVisible = true, x = x, y = y)
    }
}
