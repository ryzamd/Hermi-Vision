package com.hermitech.hermivision.domain.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import com.hermitech.hermivision.data.model.BallFrame
import java.nio.FloatBuffer
import kotlin.math.ceil
import kotlin.math.roundToInt
import kotlinx.coroutines.channels.ReceiveChannel
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.imgproc.Imgproc

class TrackNetInferencer(private val sessionManager: OnnxSessionManager) {

    companion object {
        private const val ASSET_MODEL_NAME = "TrackNetV3.onnx"
        /**
         * TrackNetV3 normally uses 9 frames (3 past, 3 current, 3 future implies sequence length).
         */
        const val SEQ_LEN = 9
        private const val WIDTH = 512
        private const val HEIGHT = 288
        private const val HEATMAP_THRESHOLD = 127.0
    }

    private var session: OrtSession? = null
    private val weight = getEnsembleWeight(SEQ_LEN)
    private val inputBuffer = FloatBuffer.allocate(1 * SEQ_LEN * 3 * HEIGHT * WIDTH)
    private val frameBuffer = Array(SEQ_LEN) { Array(3) { FloatArray(HEIGHT * WIDTH) } }

    fun loadModel() {
        session = sessionManager.loadSession(ASSET_MODEL_NAME)
    }

    fun releaseModel() {
        session?.close()
        session = null
    }

    suspend fun inferFromChannel(inputChannel: ReceiveChannel<Mat>, origWidth: Int, origHeight: Int, onProgress: (Int) -> Unit = {}): List<BallFrame> {
        val wScaler = origWidth.toFloat() / WIDTH
        val hScaler = origHeight.toFloat() / HEIGHT
        val sess = session ?: throw IllegalStateException("Model not loaded. Call loadModel() first.")
        val results = ArrayList<BallFrame>()
        val bufferSize = SEQ_LEN - 1
        val zeroPad = FloatArray(SEQ_LEN * HEIGHT * WIDTH)

        // Heatmap buffer for temporal ensembling
        val heatmapBuffer = ArrayDeque<FloatArray>(SEQ_LEN + bufferSize)
        // Sliding window of preprocessed frames
        val slidingWindow = ArrayDeque<Array<FloatArray>>(SEQ_LEN)

        repeat(bufferSize) { heatmapBuffer.addLast(zeroPad) }

        var sampleCount = 0
        var totalInputFrames = 0

        val channelMats = ArrayList<Mat>(3)

        try {
            // Consume frames from the hardware decoder output channel
            for (mat in inputChannel) {
                val targetArrays = frameBuffer[totalInputFrames % SEQ_LEN]
                totalInputFrames++

                // 1. Resize and normalize directly without new FloatArray allocations
                val resized = Mat()
                Imgproc.resize(
                        mat,
                        resized,
                        org.opencv.core.Size(WIDTH.toDouble(), HEIGHT.toDouble())
                )
                mat.release()

                val floatMat = Mat()
                resized.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0)
                resized.release()

                org.opencv.core.Core.split(floatMat, channelMats)
                floatMat.release()

                for (i in 0 until 3) {
                    channelMats[i].get(0, 0, targetArrays[i])
                    channelMats[i].release()
                }
                channelMats.clear()

                slidingWindow.addLast(targetArrays)

                // Wait until the sliding window is full
                if (slidingWindow.size < SEQ_LEN) {
                    continue
                }

                // Run ONNX inference
                val rawOutput = runSingleBatchFromChannels(sess, slidingWindow)
                heatmapBuffer.addLast(rawOutput)

                // Ensemble current frame using heatmaps in buffer
                val ensembled = FloatArray(HEIGHT * WIDTH)
                if (sampleCount < bufferSize) {
                    for (k in 0 until SEQ_LEN) {
                        val framePos = SEQ_LEN - 1 - k
                        addSlice(ensembled, heatmapBuffer.elementAt(k), framePos)
                    }
                    divideInPlace(ensembled, sampleCount + 1)
                } else {
                    for (k in 0 until SEQ_LEN) {
                        val framePos = SEQ_LEN - 1 - k
                        addWeightedSlice(ensembled, heatmapBuffer.elementAt(k), framePos, weight[k])
                    }
                }
                results.add(heatmapToBallFrame(ensembled, sampleCount, wScaler, hScaler))
                sampleCount++
                onProgress(sampleCount)

                // Shift windows
                slidingWindow.removeFirst()
                while (heatmapBuffer.size > bufferSize) {
                    heatmapBuffer.removeFirst()
                }
            }

            // Handle the tail (finishing remaining frames after video ends)
            if (totalInputFrames > 0 && sampleCount > 0) {
                repeat(bufferSize) { heatmapBuffer.addLast(zeroPad) }
                for (f in 1 until SEQ_LEN) {
                    val tailEnsembled = FloatArray(HEIGHT * WIDTH)
                    for (k in 0 until SEQ_LEN) {
                        val framePos = SEQ_LEN - 1 - k
                        if (k + f < heatmapBuffer.size) {
                            addSlice(tailEnsembled, heatmapBuffer.elementAt(k + f), framePos)
                        }
                    }
                    divideInPlace(tailEnsembled, SEQ_LEN - f)
                    results.add(heatmapToBallFrame(tailEnsembled, sampleCount, wScaler, hScaler))
                    sampleCount++
                    onProgress(sampleCount)
                }
            }
        } catch (e: Exception) {
            // Ensure cleanup on exception
            throw e
        }

        return results
    }

    /** Run model from pre-cached channel data, reusing the pre-allocated FloatBuffer. */
    private fun runSingleBatchFromChannels(
            session: OrtSession,
            windowChannels: Collection<Array<FloatArray>>
    ): FloatArray {
        inputBuffer.rewind()
        for (channels in windowChannels) {
            for (ch in channels) {
                inputBuffer.put(ch)
            }
        }
        inputBuffer.rewind()

        val shape = longArrayOf(1, (SEQ_LEN * 3).toLong(), HEIGHT.toLong(), WIDTH.toLong())
        OnnxTensor.createTensor(sessionManager.env, inputBuffer, shape).use { inputTensor ->
            session.run(mapOf("input" to inputTensor)).use { result ->
                @Suppress("UNCHECKED_CAST") val outputArray = result[0].value
                return flattenOutput(outputArray)
            }
        }
    }

    private fun heatmapToBallFrame(
            heatmap: FloatArray,
            frameId: Int,
            wScaler: Float,
            hScaler: Float
    ): BallFrame {
        val binary = Mat(HEIGHT, WIDTH, CvType.CV_8UC1)
        for (r in 0 until HEIGHT) {
            for (c in 0 until WIDTH) {
                val v = heatmap[r * WIDTH + c]
                binary.put(r, c, if (v > HEATMAP_THRESHOLD) 255.0 else 0.0)
            }
        }

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(
                binary,
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE
        )
        hierarchy.release()
        binary.release()

        if (contours.isEmpty()) {
            contours.forEach { it.release() }
            return BallFrame(frameId, isVisible = false, x = null, y = null)
        }

        var maxArea = 0.0
        var bestRect = org.opencv.core.Rect()
        for (cnt in contours) {
            val rect = Imgproc.boundingRect(cnt)
            val area = rect.width.toDouble() * rect.height
            if (area > maxArea) {
                maxArea = area
                bestRect = rect
            }
            cnt.release()
        }

        val cx = ((bestRect.x + bestRect.width / 2.0) * wScaler).roundToInt().toFloat()
        val cy = ((bestRect.y + bestRect.height / 2.0) * hScaler).roundToInt().toFloat()

        val isVisible = cx != 0f || cy != 0f
        return BallFrame(
                frameId,
                isVisible,
                if (isVisible) cx else null,
                if (isVisible) cy else null
        )
    }

    private fun getEnsembleWeight(seqLen: Int): FloatArray {
        val w = FloatArray(seqLen) { 1f }
        for (i in 0 until ceil(seqLen / 2.0).toInt()) {
            w[i] = (i + 1).toFloat()
            w[seqLen - i - 1] = (i + 1).toFloat()
        }
        val sum = w.sum()
        for (i in w.indices) w[i] /= sum
        return w
    }

    private fun addSlice(dst: FloatArray, src: FloatArray, frameIdx: Int) {
        val offset = frameIdx * HEIGHT * WIDTH
        for (i in dst.indices) dst[i] += src[offset + i]
    }

    private fun addWeightedSlice(dst: FloatArray, src: FloatArray, frameIdx: Int, weight: Float) {
        val offset = frameIdx * HEIGHT * WIDTH
        for (i in dst.indices) dst[i] += src[offset + i] * weight
    }

    private fun divideInPlace(arr: FloatArray, divisor: Int) {
        val d = divisor.toFloat()
        for (i in arr.indices) arr[i] /= d
    }

    @Suppress("UNCHECKED_CAST")
    private fun flattenOutput(output: Any): FloatArray {
        return when (output) {
            is Array<*> -> {
                val batch = output as Array<Array<Array<FloatArray>>>
                val seq = batch[0]
                val result = FloatArray(SEQ_LEN * HEIGHT * WIDTH)
                var pos = 0
                for (s in seq.indices) {
                    for (h in seq[s].indices) {
                        System.arraycopy(seq[s][h], 0, result, pos, seq[s][h].size)
                        pos += seq[s][h].size
                    }
                }
                result
            }
            else -> throw IllegalStateException("Unexpected ONNX output type: ${output::class}")
        }
    }
}
