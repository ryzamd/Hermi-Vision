package com.hermitech.hermivision.domain.inference

import ai.onnxruntime.OnnxTensor
import com.hermitech.hermivision.data.model.BallFrame
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Bounce Detection engine.
 *
 * Port of TennisProject/bounce_detector.py.
 *
 * Pipeline:
 *  1. smooth_predictions → extrapolate missing ball positions via natural cubic spline
 *  2. prepare_features   → compute 12 lag/diff/div features per frame
 *  3. ONNX predict       → CatBoost ONNX-ML model (300 trees, 12 numerical features)
 *  4. threshold(0.45)    → identify bounce candidates
 *  5. postprocess         → merge adjacent detections, keep peak score
 *
 * Feature order (must match training):
 *   [x_diff_1, x_diff_2, x_diff_inv_1, x_diff_inv_2, x_div_1, x_div_2,
 *    y_diff_1, y_diff_2, y_diff_inv_1, y_diff_inv_2, y_div_1, y_div_2]
 *
 * ONNX-ML note:
 *   CatBoost exported as ONNX-ML uses `ai_onnx_ml` domain operators
 *   (TreeEnsembleRegressor). onnxruntime-android supports this natively.
 *   Input name: "features" (float tensor [N, 12])
 *   Output: float tensor [N] (regression scores)
 */
class BounceDetectorInferencer(private val sessionManager: OnnxSessionManager) {

    companion object {
        const val MODEL_NAME = "BounceDetector.onnx"

        const val BOUNCE_THRESHOLD = 0.45f

        private const val INTERP = 5
        private const val MAX_CONSECUTIVE_EXTRAP = 3
        private const val OUTLIER_DIST = 80.0f
        private const val EPS = 1e-15f
        private const val NUM_LAGS = 3
    }

    private var cachedSession: ai.onnxruntime.OrtSession? = null

    /**
     * Detect bounce frames from ball trajectory.
     *
     * Port of BounceDetector.predict().
     *
     * @param ballFrames   Full trajectory from TrackNet inference
     * @param smooth       Whether to run smooth_predictions first (default true)
     * @return Set of frame IDs where bounces were detected
     */
    fun detectBounces(ballFrames: List<BallFrame>, smooth: Boolean = true): Set<Int> {
        if (ballFrames.size < INTERP + NUM_LAGS) return emptySet()

        val xBall = ballFrames.map { it.x }.toMutableList()
        val yBall = ballFrames.map { it.y }.toMutableList()

        if (smooth) {
            smoothPredictions(xBall, yBall)
        }

        val (features, frameIndices) = prepareFeatures(xBall, yBall)
        if (features.isEmpty()) return emptySet()

        val predictions = runOnnxInference(features)

        val indBounce = buildList {
            for (idx in predictions.indices) {
                if (predictions[idx] > BOUNCE_THRESHOLD) add(idx)
            }
        }

        if (indBounce.isEmpty()) return emptySet()

        val filtered = postprocess(indBounce, predictions)

        return filtered.map { frameIndices[it] }.toSet()
    }

    /**
     * Port of BounceDetector.smooth_predictions().
     *
     * Extrapolates up to [MAX_CONSECUTIVE_EXTRAP] consecutive missing frames
     * using natural cubic spline on the last [INTERP] valid points.
     * After extrapolation, checks distance to next real point — if > 80px,
     * nullifies the next point as outlier.
     */
    private fun smoothPredictions(xBall: MutableList<Float?>, yBall: MutableList<Float?>) {
        val isNone = xBall.map { if (it == null) 1 else 0 }.toMutableList()
        var counter = 0

        for (num in INTERP until xBall.size - 1) {
            val isCurrentNull = xBall[num] == null
            val prevAllValid = (num - INTERP until num).sumOf { isNone[it] } == 0

            if (isCurrentNull && prevAllValid && counter < MAX_CONSECUTIVE_EXTRAP) {
                val xWindow = (num - INTERP until num).map { xBall[it]!! }
                val yWindow = (num - INTERP until num).map { yBall[it]!! }

                val (xExt, yExt) = extrapolate(xWindow, yWindow)
                xBall[num] = xExt
                yBall[num] = yExt
                isNone[num] = 0

                if (num + 1 < xBall.size && xBall[num + 1] != null) {
                    val dist = euclideanDist(xExt, yExt, xBall[num + 1]!!, yBall[num + 1]!!)
                    if (dist > OUTLIER_DIST) {
                        xBall[num + 1] = null
                        yBall[num + 1] = null
                        isNone[num + 1] = 1
                    }
                }
                counter++
            } else {
                counter = 0
            }
        }
    }

    /**
     * Port of BounceDetector.extrapolate() — natural cubic spline.
     *
     * Python: CubicSpline(xs, coords, bc_type='natural')
     * Extrapolates 1 point beyond the given window.
     */
    private fun extrapolate(xCoords: List<Float>, yCoords: List<Float>): Pair<Float, Float> {
        val xs = FloatArray(xCoords.size) { it.toFloat() }
        val nextX = xCoords.size.toFloat()

        val xExt = naturalCubicSplineEval(xs, xCoords.toFloatArray(), nextX)
        val yExt = naturalCubicSplineEval(xs, yCoords.toFloatArray(), nextX)
        return Pair(xExt, yExt)
    }

    /**
     * Port of BounceDetector.prepare_features().
     *
     * Computes 12 features per frame and returns only valid rows
     * (where all lags are non-null and x-coordinate is non-null).
     *
     * Feature order (MUST match CatBoost training order):
     *   x_diff_1, x_diff_2, x_diff_inv_1, x_diff_inv_2, x_div_1, x_div_2,
     *   y_diff_1, y_diff_2, y_diff_inv_1, y_diff_inv_2, y_div_1, y_div_2
     *
     * @return Pair(features list of 12-element arrays, corresponding frame indices)
     */
    private fun prepareFeatures(xBall: List<Float?>, yBall: List<Float?>): Pair<List<FloatArray>, List<Int>> {
        val n = xBall.size
        val features = mutableListOf<FloatArray>()
        val frameIndices = mutableListOf<Int>()

        for (i in 0 until n) {
            val xCurr = xBall[i] ?: continue
            val yCurr = yBall[i] ?: continue

            var allLagsValid = true
            for (lag in 1 until NUM_LAGS) {
                if (i - lag < 0 || xBall[i - lag] == null) { allLagsValid = false; break }
                if (i + lag >= n || xBall[i + lag] == null) { allLagsValid = false; break }
            }
            if (!allLagsValid) continue

            val featureRow = FloatArray(12)

            val xDiff1 = abs(xBall[i - 1]!! - xCurr)
            // x_diff_2 = abs(x_lag_2 - x)
            val xDiff2 = abs(xBall[i - 2]!! - xCurr)
            // x_diff_inv_1 = abs(x_lag_inv_1 - x)   where x_lag_inv_1 = x[i+1]
            val xDiffInv1 = abs(xBall[i + 1]!! - xCurr)
            // x_diff_inv_2 = abs(x_lag_inv_2 - x)
            val xDiffInv2 = abs(xBall[i + 2]!! - xCurr)
            // x_div_1 = abs(x_diff_1 / (x_diff_inv_1 + eps))
            val xDiv1 = abs(xDiff1 / (xDiffInv1 + EPS))
            // x_div_2 = abs(x_diff_2 / (x_diff_inv_2 + eps))
            val xDiv2 = abs(xDiff2 / (xDiffInv2 + EPS))

            // --- Y features (NO abs for diff, NO abs for div) ---
            // y_diff_1 = y_lag_1 - y   (Python: labels['y_lag_{}'.format(i)] - labels['y-coordinate'])
            val yDiff1 = yBall[i - 1]!! - yCurr
            // y_diff_2 = y_lag_2 - y
            val yDiff2 = yBall[i - 2]!! - yCurr
            // y_diff_inv_1 = y_lag_inv_1 - y
            val yDiffInv1 = yBall[i + 1]!! - yCurr
            // y_diff_inv_2 = y_lag_inv_2 - y
            val yDiffInv2 = yBall[i + 2]!! - yCurr
            // y_div_1 = y_diff_1 / (y_diff_inv_1 + eps)   (NO abs)
            val yDiv1 = yDiff1 / (yDiffInv1 + EPS)
            // y_div_2 = y_diff_2 / (y_diff_inv_2 + eps)
            val yDiv2 = yDiff2 / (yDiffInv2 + EPS)

            // Order: [x_diff_1, x_diff_2, x_diff_inv_1, x_diff_inv_2, x_div_1, x_div_2,
            //          y_diff_1, y_diff_2, y_diff_inv_1, y_diff_inv_2, y_div_1, y_div_2]
            featureRow[0]  = xDiff1
            featureRow[1]  = xDiff2
            featureRow[2]  = xDiffInv1
            featureRow[3]  = xDiffInv2
            featureRow[4]  = xDiv1
            featureRow[5]  = xDiv2
            featureRow[6]  = yDiff1
            featureRow[7]  = yDiff2
            featureRow[8]  = yDiffInv1
            featureRow[9]  = yDiffInv2
            featureRow[10] = yDiv1
            featureRow[11] = yDiv2

            features.add(featureRow)
            frameIndices.add(i)  // original frame index
        }

        return Pair(features, frameIndices)
    }

    /**
     * Run CatBoost ONNX-ML inference.
     *
     * CatBoost ONNX-ML models use TreeEnsembleRegressor from the ai.onnx.ml domain.
     * Input: float tensor [N, 12]
     * Output: float tensor [N] — regression scores
     */
    private fun runOnnxInference(features: List<FloatArray>): FloatArray {
        val numSamples = features.size
        val numFeatures = 12
        val flatBuffer = FloatBuffer.allocate(numSamples * numFeatures)

        for (row in features) {
            flatBuffer.put(row)
        }
        flatBuffer.rewind()

        val session = getCachedSession()
        try {
            val inputName = session.inputNames.first()

            OnnxTensor.createTensor(
                sessionManager.env,
                flatBuffer,
                longArrayOf(numSamples.toLong(), numFeatures.toLong())
            ).use { inputTensor ->
                session.run(mapOf(inputName to inputTensor)).use { result ->
                    val outputValue = result[0].value

                    return when (outputValue) {
                        is FloatArray -> outputValue
                        is Array<*> -> {
                            @Suppress("UNCHECKED_CAST")
                            val arr = outputValue as? Array<FloatArray>
                            if (arr != null) {
                                FloatArray(arr.size) { arr[it][0] }
                            } else {
                                FloatArray(numSamples) { 0f }
                            }
                        }
                        else -> FloatArray(numSamples) { 0f }
                    }
                }
            }
        }
    }

    private fun getCachedSession(): ai.onnxruntime.OrtSession {
        return cachedSession ?: sessionManager.loadSession(MODEL_NAME).also {
            cachedSession = it
        }
    }

    fun releaseSession() {
        cachedSession?.close()
        cachedSession = null
    }

    /**
     * Port of BounceDetector.postprocess().
     *
     * Merge adjacent bounce indices: if two consecutive indices differ by 1,
     * keep the one with the higher prediction score.
     *
     * @param indBounce  Sorted indices into the predictions array where score > threshold
     * @param preds      Full predictions array
     * @return Filtered list of indices
     */
    private fun postprocess(indBounce: List<Int>, preds: FloatArray): List<Int> {
        if (indBounce.isEmpty()) return emptyList()

        val filtered = mutableListOf(indBounce[0])

        for (i in 1 until indBounce.size) {
            val currIdx = indBounce[i]
            val prevIdx = indBounce[i - 1]

            if (currIdx - prevIdx != 1) {
                filtered.add(currIdx)
            } else if (preds[currIdx] > preds[prevIdx]) {
                filtered[filtered.size - 1] = currIdx
            }
            // else: adjacent but previous score is higher → keep previous (do nothing)
        }

        return filtered
    }

    /**
     * Evaluate a natural cubic spline at a single point.
     *
     * Implements CubicSpline(xs, ys, bc_type='natural') from SciPy.
     * Natural boundary conditions: S''(x_0) = 0 and S''(x_n) = 0.
     *
     * Used only for extrapolation of 5 points → 1 next point.
     * This is a minimal implementation sufficient for our use case.
     *
     * @param xs  Knot x-coordinates (must be sorted ascending, equally spaced here)
     * @param ys  Knot y-values
     * @param t   Point to evaluate at (can be outside the knot range for extrapolation)
     * @return    Interpolated/extrapolated value at t
     */
    private fun naturalCubicSplineEval(xs: FloatArray, ys: FloatArray, t: Float): Float {
        val n = xs.size
        if (n < 2) return ys.lastOrNull() ?: 0f
        if (n == 2) {
            // Linear interpolation/extrapolation
            val slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
            return ys[0] + slope * (t - xs[0])
        }

        // Compute intervals h[i] = x[i+1] - x[i]
        val h = FloatArray(n - 1) { xs[it + 1] - xs[it] }

        // Set up tridiagonal system for second derivatives c[i]
        // Natural BC: c[0] = 0, c[n-1] = 0
        // For interior points (i = 1..n-2):
        //   h[i-1]*c[i-1] + 2*(h[i-1]+h[i])*c[i] + h[i]*c[i+1]
        //     = 3*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
        val c = FloatArray(n) // Second derivatives / 6 actually cubic coefficients

        if (n > 2) {
            // Solve tridiagonal system using Thomas algorithm
            val interiorSize = n - 2
            val lower = FloatArray(interiorSize)    // sub-diagonal
            val diag = FloatArray(interiorSize)     // main diagonal
            val upper = FloatArray(interiorSize)    // super-diagonal
            val rhs = FloatArray(interiorSize)      // right-hand side

            for (i in 0 until interiorSize) {
                val ii = i + 1  // actual index in original arrays
                lower[i] = h[ii - 1]
                diag[i] = 2f * (h[ii - 1] + h[ii])
                upper[i] = h[ii]
                rhs[i] = 3f * ((ys[ii + 1] - ys[ii]) / h[ii] - (ys[ii] - ys[ii - 1]) / h[ii - 1])
            }

            // Forward elimination
            for (i in 1 until interiorSize) {
                val factor = lower[i] / diag[i - 1]
                diag[i] -= factor * upper[i - 1]
                rhs[i] -= factor * rhs[i - 1]
            }

            // Back substitution
            val cInterior = FloatArray(interiorSize)
            cInterior[interiorSize - 1] = rhs[interiorSize - 1] / diag[interiorSize - 1]
            for (i in interiorSize - 2 downTo 0) {
                cInterior[i] = (rhs[i] - upper[i] * cInterior[i + 1]) / diag[i]
            }

            // Map back: c[0] = 0 (natural BC), c[n-1] = 0 (natural BC)
            for (i in 0 until interiorSize) {
                c[i + 1] = cInterior[i]
            }
        }

        // Compute polynomial coefficients for each interval
        // S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3
        val a = ys.copyOf()
        val b = FloatArray(n - 1)
        val d = FloatArray(n - 1)

        for (i in 0 until n - 1) {
            b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (2f * c[i] + c[i + 1]) / 3f
            d[i] = (c[i + 1] - c[i]) / (3f * h[i])
        }

        // Evaluate at t
        // Find the interval [x_k, x_{k+1}] that t falls in (or extrapolate from last)
        val k = if (t <= xs[0]) {
            0
        } else if (t >= xs[n - 1]) {
            n - 2  // Use last interval for extrapolation
        } else {
            var idx = 0
            for (i in 0 until n - 1) {
                if (t >= xs[i] && t < xs[i + 1]) {
                    idx = i
                    break
                }
            }
            idx
        }

        val dx = t - xs[k]
        return a[k] + b[k] * dx + c[k] * dx * dx + d[k] * dx * dx * dx
    }

    private fun euclideanDist(x1: Float, y1: Float, x2: Float, y2: Float): Float {
        val dx = x1 - x2
        val dy = y1 - y2
        return sqrt(dx * dx + dy * dy)
    }
}