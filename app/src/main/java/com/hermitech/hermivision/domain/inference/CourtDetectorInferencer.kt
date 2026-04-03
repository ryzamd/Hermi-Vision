package com.hermitech.hermivision.domain.inference

import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import com.hermitech.hermivision.data.model.CourtMatrix
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.sqrt

/**
 * Court Detector ONNX inference engine.
 *
 * Port of TennisCourtDetector/infer_in_video.py + postprocess.py + homography.py.
 *
 * Input shape:  float32[1, 3, 360, 640]   — 1 RGB frame
 * Output shape: float32[1, 15, 360, 640]  — 15 heatmaps (14 keypoints + 1 extra)
 *
 * Pipeline per frame:
 *  1. Resize → 640×360, normalise [0,1], CHW
 *  2. Inference → 15 heatmaps
 *  3. For each of 14 heatmaps: threshold → HoughCircles → (x,y)
 *  4. Optional: refine_kps via HoughLinesP intersection
 *  5. Try 12 homography configurations → pick best → invert matrix
 *  6. Return CourtMatrix with homographyInv (3×3 = 9 floats)
 */
class CourtDetectorInferencer(private val sessionManager: OnnxSessionManager) {
    companion object {
        const val MODEL_NAME = "CourtDetector.onnx"

        const val INPUT_WIDTH  = 640
        const val INPUT_HEIGHT = 360
        const val NUM_KEYPOINTS = 14
        const val HEATMAP_LOW_THRESH = 170.0
        const val MIN_RADIUS = 10
        const val MAX_RADIUS = 25
        const val SCALE = 2
        const val REFINE_CROP_SIZE = 40
        val SKIP_REFINE = setOf(8, 9, 12)
    }

    private val courtFloatBuffer = FloatBuffer.allocate(3 * INPUT_HEIGHT * INPUT_WIDTH)
    private val courtFloatArray = FloatArray(INPUT_HEIGHT * INPUT_WIDTH)
    private var session: OrtSession? = null

    fun loadModel() {
        session = sessionManager.loadSession(MODEL_NAME)
    }

    fun releaseModel() {
        session?.close()
        session = null
    }

    /**
     * Run court detection on a single frame.
     *
     * @param frame   Original-resolution bitmap.
     * @param frameId Index of the frame in the video.
     * @param useRefine  Apply line-intersection refinement.
     * @return CourtMatrix with homographyInv (null if court not detected).
     */
    fun inferSingleFrame(frame: Bitmap, frameId: Int, useRefine: Boolean = true): CourtMatrix {
        val sess = session ?: throw IllegalStateException("Model not loaded.")

        val inputBuffer = preprocessFrame(frame)
        val shape = longArrayOf(1, 3, INPUT_HEIGHT.toLong(), INPUT_WIDTH.toLong())

        val points: Array<FloatArray?> // 14 detected keypoints, nullable per kp

        OnnxTensor.createTensor(sessionManager.env, inputBuffer, shape).use { inputTensor ->
            sess.run(mapOf("input" to inputTensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val output = result[0].value
                points = extractKeypoints(output, frame, useRefine)
            }
        }

        // Compute homography
        val homographyInv = computeBestHomography(points)

        return CourtMatrix(
            frameId = frameId,
            homographyInv = homographyInv
        )
    }

    /**
     * Run court detection consuming a channel of OpenCV Mat frames.
     * The consumer is responsible for releasing the Mats.
     */
    suspend fun inferFromChannel(
        inputChannel: kotlinx.coroutines.channels.ReceiveChannel<Mat>,
        useRefine: Boolean = true,
        onProgress: ((Int) -> Unit)? = null
    ): List<CourtMatrix> = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Default) {
        val sess = session ?: throw IllegalStateException("Model not loaded.")
        val results = mutableListOf<CourtMatrix>()
        var frameId = 0

        for (frameMat in inputChannel) {
            val inputBuffer = preprocessFrameMat(frameMat)
            val shape = longArrayOf(1, 3, INPUT_HEIGHT.toLong(), INPUT_WIDTH.toLong())

            val points: Array<FloatArray?>

            OnnxTensor.createTensor(sessionManager.env, inputBuffer, shape).use { inputTensor ->
                sess.run(mapOf("input" to inputTensor)).use { result ->
                    @Suppress("UNCHECKED_CAST")
                    val output = result[0].value
                    points = extractKeypointsMat(output, frameMat, useRefine)
                }
            }

            val homographyInv = computeBestHomography(points)
            results.add(CourtMatrix(frameId = frameId, homographyInv = homographyInv))

            frameMat.release()
            frameId++
            onProgress?.invoke(frameId)
        }

        results
    }

    private fun preprocessFrame(bitmap: Bitmap): FloatBuffer {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
        Imgproc.resize(mat, mat, Size(INPUT_WIDTH.toDouble(), INPUT_HEIGHT.toDouble()))

        val floatMat = Mat()
        mat.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0)
        mat.release()

        courtFloatBuffer.rewind()
        val channels = ArrayList<Mat>(3)
        Core.split(floatMat, channels)
        for (ch in channels) {
            ch.get(0, 0, courtFloatArray)
            courtFloatBuffer.put(courtFloatArray)
            ch.release()
        }
        floatMat.release()

        courtFloatBuffer.rewind()
        return courtFloatBuffer
    }

    private fun preprocessFrameMat(origMatRgb: Mat): FloatBuffer {
        val mat = Mat()
        Imgproc.resize(origMatRgb, mat, Size(INPUT_WIDTH.toDouble(), INPUT_HEIGHT.toDouble()))

        val floatMat = Mat()
        mat.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0)
        mat.release()

        courtFloatBuffer.rewind()
        val channels = ArrayList<Mat>(3)
        Core.split(floatMat, channels)
        for (ch in channels) {
            ch.get(0, 0, courtFloatArray)
            courtFloatBuffer.put(courtFloatArray)
            ch.release()
        }
        floatMat.release()

        courtFloatBuffer.rewind()
        return courtFloatBuffer
    }

    /**
     * From the model output [1, 15, H, W], extract 14 keypoints.
     */
    @Suppress("UNCHECKED_CAST")
    private fun extractKeypoints(rawOutput: Any, originalFrame: Bitmap, useRefine: Boolean): Array<FloatArray?> {
        val pred = flattenTo3D(rawOutput)
        val points = arrayOfNulls<FloatArray>(NUM_KEYPOINTS)

        val origMat = if (useRefine) {
            val m = Mat()
            Utils.bitmapToMat(originalFrame, m)
            Imgproc.cvtColor(m, m, Imgproc.COLOR_RGBA2BGR)
            m
        } else null

        for (kp in 0 until NUM_KEYPOINTS) {
            val heatmap = Mat(INPUT_HEIGHT, INPUT_WIDTH, CvType.CV_8UC1)
            for (r in 0 until INPUT_HEIGHT) {
                for (c in 0 until INPUT_WIDTH) {
                    val raw = pred[kp][r * INPUT_WIDTH + c]
                    // ONNX model outputs raw logits — apply sigmoid to get [0,1]
                    val sigmoided = 1.0f / (1.0f + kotlin.math.exp(-raw))
                    val v = (sigmoided * 255).toInt().coerceIn(0, 255)
                    heatmap.put(r, c, v.toDouble())
                }
            }

            val (xPred, yPred) = postprocessHeatmap(heatmap)
            heatmap.release()

            if (xPred != null && yPred != null) {
                if (useRefine && kp !in SKIP_REFINE && origMat != null) {
                    val (rx, ry) = refineKeypoint(origMat, yPred.toInt(), xPred.toInt())
                    points[kp] = floatArrayOf(rx, ry)
                } else {
                    points[kp] = floatArrayOf(xPred, yPred)
                }
            }
        }

        origMat?.release()
        return points
    }

    @Suppress("UNCHECKED_CAST")
    private fun extractKeypointsMat(rawOutput: Any, originalFrameRgb: Mat, useRefine: Boolean): Array<FloatArray?> {
        val pred = flattenTo3D(rawOutput)   // [15, H, W]
        val points = arrayOfNulls<FloatArray>(NUM_KEYPOINTS)

        val origMatBgr = if (useRefine) {
            val m = Mat()
            Imgproc.cvtColor(originalFrameRgb, m, Imgproc.COLOR_RGB2BGR)
            m
        } else null

        for (kp in 0 until NUM_KEYPOINTS) {
            val heatmap = Mat(INPUT_HEIGHT, INPUT_WIDTH, CvType.CV_8UC1)
            for (r in 0 until INPUT_HEIGHT) {
                for (c in 0 until INPUT_WIDTH) {
                    val raw = pred[kp][r * INPUT_WIDTH + c]
                    val sigmoided = 1.0f / (1.0f + kotlin.math.exp(-raw))
                    val v = (sigmoided * 255).toInt().coerceIn(0, 255)
                    heatmap.put(r, c, v.toDouble())
                }
            }

            val (xPred, yPred) = postprocessHeatmap(heatmap)
            heatmap.release()

            if (xPred != null && yPred != null) {
                if (useRefine && kp !in SKIP_REFINE && origMatBgr != null) {
                    val (rx, ry) = refineKeypoint(origMatBgr, yPred.toInt(), xPred.toInt())
                    points[kp] = floatArrayOf(rx, ry)
                } else {
                    points[kp] = floatArrayOf(xPred, yPred)
                }
            }
        }

        origMatBgr?.release()
        return points
    }

    /**
     * Port of postprocess.py postprocess().
     * threshold → HoughCircles → first circle centre × scale.
     */
    private fun postprocessHeatmap(heatmap: Mat): Pair<Float?, Float?> {
        val binary = Mat()
        Imgproc.threshold(heatmap, binary, HEATMAP_LOW_THRESH, 255.0, Imgproc.THRESH_BINARY)

        val circles = Mat()
        Imgproc.HoughCircles(
            binary, circles, Imgproc.HOUGH_GRADIENT,
            1.0, 20.0, 50.0, 2.0,
            MIN_RADIUS, MAX_RADIUS
        )
        binary.release()

        return if (circles.cols() > 0) {
            val cx = circles.get(0, 0)[0].toFloat() * SCALE
            val cy = circles.get(0, 0)[1].toFloat() * SCALE
            circles.release()
            Pair(cx, cy)
        } else {
            circles.release()
            Pair(null, null)
        }
    }

    /**
     * Port of postprocess.py refine_kps().
     * Crop around detected point, detect lines, find intersection.
     */
    private fun refineKeypoint(image: Mat, yCenter: Int, xCenter: Int): Pair<Float, Float> {
        val imgH = image.rows()
        val imgW = image.cols()
        val xMin = (yCenter - REFINE_CROP_SIZE).coerceAtLeast(0)
        val xMax = (yCenter + REFINE_CROP_SIZE).coerceAtMost(imgH)
        val yMin = (xCenter - REFINE_CROP_SIZE).coerceAtLeast(0)
        val yMax = (xCenter + REFINE_CROP_SIZE).coerceAtMost(imgW)

        val crop = image.submat(xMin, xMax, yMin, yMax)

        val gray = Mat()
        Imgproc.cvtColor(crop, gray, Imgproc.COLOR_BGR2GRAY)
        val binaryGray = Mat()
        Imgproc.threshold(gray, binaryGray, 155.0, 255.0, Imgproc.THRESH_BINARY)
        gray.release()

        val lines = Mat()
        Imgproc.HoughLinesP(binaryGray, lines, 1.0, Math.PI / 180.0, 30, 10.0, 30.0)
        binaryGray.release()
        crop.release()

        if (lines.rows() >= 2) {
            val lineList = (0 until lines.rows()).map { lines.get(it, 0) }
            val merged = mergeLines(lineList)
            lines.release()

            if (merged.size >= 2) {
                val inter = lineIntersection(merged[0], merged[1])
                if (inter != null) {
                    val newY = inter.first.toInt()
                    val newX = inter.second.toInt()
                    if (newY in 0 until (xMax - xMin) && newX in 0 until (yMax - yMin)) {
                        return Pair((yMin + newX).toFloat(), (xMin + newY).toFloat())
                    }
                }
            }
        } else {
            lines.release()
        }

        return Pair(xCenter.toFloat(), yCenter.toFloat())
    }

    /**
     * Try all 12 court configurations, find best homography.
     * Returns the INVERSE matrix flattened to 9 floats, or null.
     */
    private fun computeBestHomography(detectedPoints: Array<FloatArray?>): FloatArray? {
        val referKps = CourtReference.keyPoints
        var bestMatrix: Mat? = null
        var bestDist = Float.MAX_VALUE

        for (confId in 1..12) {
            val confRef = CourtReference.courtConf[confId] ?: continue
            val confIndices = CourtReference.courtConfIndices[confId] ?: continue

            // Check all 4 required keypoints are detected
            val detected4 = confIndices.map { detectedPoints[it] }
            if (detected4.any { it == null }) continue

            val srcPts = MatOfPoint2f(
                *confRef.map { Point(it[0].toDouble(), it[1].toDouble()) }.toTypedArray()
            )
            val dstPts = MatOfPoint2f(
                *detected4.map { Point(it!![0].toDouble(), it[1].toDouble()) }.toTypedArray()
            )

            val matrix = Calib3d.findHomography(srcPts, dstPts, 0)
            srcPts.release()
            dstPts.release()

            if (matrix.empty()) {
                matrix.release()
                continue
            }

            // Transform all 14 reference keypoints by this matrix and measure error
            val refMat = MatOfPoint2f(
                *referKps.map { Point(it[0].toDouble(), it[1].toDouble()) }.toTypedArray()
            )
            val transformed = MatOfPoint2f()
            Core.perspectiveTransform(refMat, transformed, matrix)
            refMat.release()

            var totalDist = 0.0
            var count = 0
            for (i in 0 until 12) {  // use first 12 keypoints for error measurement
                if (i in confIndices.toSet()) continue
                val dp = detectedPoints[i] ?: continue
                val tp = transformed.get(i, 0)
                val d = sqrt(
                    (dp[0] - tp[0]).let { it * it } +
                    (dp[1] - tp[1]).let { it * it }
                )
                totalDist += d
                count++
            }
            transformed.release()

            val meanDist = if (count > 0) (totalDist / count).toFloat() else Float.MAX_VALUE
            if (meanDist < bestDist) {
                bestMatrix?.release()
                bestMatrix = matrix
                bestDist = meanDist
            } else {
                matrix.release()
            }
        }

        if (bestMatrix == null) return null

        // Invert the matrix
        val invMatrix = Mat()
        Core.invert(bestMatrix, invMatrix)
        bestMatrix.release()

        val result = FloatArray(9)
        for (r in 0 until 3) {
            for (c in 0 until 3) {
                result[r * 3 + c] = invMatrix.get(r, c)[0].toFloat()
            }
        }
        invMatrix.release()
        return result
    }

    private fun mergeLines(lines: List<DoubleArray>): List<DoubleArray> {
        val sorted = lines.sortedBy { it[0] }
        val mask = BooleanArray(sorted.size) { true }
        val merged = mutableListOf<DoubleArray>()

        for (i in sorted.indices) {
            if (!mask[i]) continue
            var line = sorted[i].copyOf()
            for (j in i + 1 until sorted.size) {
                if (!mask[j]) continue
                val d1 = euclidean(line[0], line[1], sorted[j][0], sorted[j][1])
                val d2 = euclidean(line[2], line[3], sorted[j][2], sorted[j][3])
                if (d1 < 20 && d2 < 20) {
                    line = doubleArrayOf(
                        (line[0] + sorted[j][0]) / 2,
                        (line[1] + sorted[j][1]) / 2,
                        (line[2] + sorted[j][2]) / 2,
                        (line[3] + sorted[j][3]) / 2
                    )
                    mask[j] = false
                }
            }
            merged.add(line)
        }
        return merged
    }

    /** Find intersection of two line segments. Returns (row, col) or null. */
    private fun lineIntersection(l1: DoubleArray, l2: DoubleArray): Pair<Double, Double>? {
        val x1 = l1[0]; val y1 = l1[1]; val x2 = l1[2]; val y2 = l1[3]
        val x3 = l2[0]; val y3 = l2[1]; val x4 = l2[2]; val y4 = l2[3]

        val denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if (kotlin.math.abs(denom) < 1e-10) return null

        val px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        val py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return Pair(py, px)     // (row, col) order matches Python convention
    }

    private fun euclidean(x1: Double, y1: Double, x2: Double, y2: Double): Double =
        sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

    /**
     * Flatten ORT output [1, 15, H, W] → [15][H*W].
     */
    @Suppress("UNCHECKED_CAST")
    private fun flattenTo3D(output: Any): Array<FloatArray> {
        return when (output) {
            is Array<*> -> {
                val batch = output as Array<Array<Array<FloatArray>>>
                val channels = batch[0]
                Array(channels.size) { ch ->
                    val flat = FloatArray(INPUT_HEIGHT * INPUT_WIDTH)
                    var pos = 0
                    for (row in channels[ch]) {
                        System.arraycopy(row, 0, flat, pos, row.size)
                        pos += row.size
                    }
                    flat
                }
            }
            else -> throw IllegalStateException("Unexpected ONNX output type: ${output::class}")
        }
    }
}