package com.hermitech.hermivision.inference

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class TrackNetOnnxDetector(
    context: Context,
    private val fallbackDetector: BallDetector = ColorBallDetector()
) : BallDetector {

    private val ortEnvironment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val ortSession: OrtSession? = runCatching {
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setInterOpNumThreads(1)
        }
        val modelBytes = context.assets.open(MODEL_PATH).use { it.readBytes() }
        ortEnvironment.createSession(modelBytes, sessionOptions)
    }.onFailure { throwable ->
        Log.e(TAG, "Failed to create TrackNet ONNX session. Falling back to color detector.", throwable)
    }.getOrNull()

    private val sequenceFrames = ArrayDeque<FloatArray>(SEQUENCE_LENGTH)
    private val backgroundFrame = FloatArray(CHANNELS_PER_FRAME * MODEL_HEIGHT * MODEL_WIDTH)
    private var backgroundInitialized = false
    private val inputBuffer = FloatArray((SEQUENCE_LENGTH + 1) * CHANNELS_PER_FRAME * MODEL_HEIGHT * MODEL_WIDTH)

    override fun detect(frameData: FrameData): RawBallCandidate? {
        val session = ortSession ?: return fallbackDetector.detect(frameData)

        val normalizedFrame = frameToModelInput(frameData)
        updateBackground(normalizedFrame)
        if (sequenceFrames.size == SEQUENCE_LENGTH) {
            sequenceFrames.removeFirst()
        }
        sequenceFrames.addLast(normalizedFrame)

        if (sequenceFrames.size < SEQUENCE_LENGTH) {
            return fallbackDetector.detect(frameData)
        }

        var offset = 0
        sequenceFrames.forEach { frame ->
            frame.copyInto(inputBuffer, destinationOffset = offset)
            offset += frame.size
        }
        backgroundFrame.copyInto(inputBuffer, destinationOffset = offset)

        return try {
            OnnxTensor.createTensor(
                ortEnvironment,
                FloatBuffer.wrap(inputBuffer),
                longArrayOf(1, ((SEQUENCE_LENGTH + 1) * CHANNELS_PER_FRAME).toLong(), MODEL_HEIGHT.toLong(), MODEL_WIDTH.toLong())
            ).use { inputTensor ->
                session.run(mapOf(INPUT_NAME to inputTensor)).use { results ->
                    val heatmaps = results[0].value as Array<Array<Array<FloatArray>>>
                    parseLatestHeatmap(heatmaps[0]) ?: fallbackDetector.detect(frameData)
                }
            }
        } catch (exception: OrtException) {
            Log.e(TAG, "TrackNet inference failed. Falling back to color detector.", exception)
            fallbackDetector.detect(frameData)
        }
    }

    override fun close() {
        sequenceFrames.clear()
        fallbackDetector.close()
        ortSession?.close()
    }

    private fun frameToModelInput(frameData: FrameData): FloatArray {
        val pixels = FloatArray(CHANNELS_PER_FRAME * MODEL_HEIGHT * MODEL_WIDTH)
        val xScale = frameData.width.toFloat() / MODEL_WIDTH
        val yScale = frameData.height.toFloat() / MODEL_HEIGHT
        val planeSize = MODEL_HEIGHT * MODEL_WIDTH

        for (y in 0 until MODEL_HEIGHT) {
            val sourceY = min((y * yScale).toInt(), frameData.height - 1)
            for (x in 0 until MODEL_WIDTH) {
                val sourceX = min((x * xScale).toInt(), frameData.width - 1)
                val sourceIndex = sourceY * frameData.rowStride + sourceX * frameData.pixelStride
                val targetIndex = y * MODEL_WIDTH + x

                val red = (frameData.rgbaBytes[sourceIndex].toInt() and 0xFF) / 255f
                val green = (frameData.rgbaBytes[sourceIndex + 1].toInt() and 0xFF) / 255f
                val blue = (frameData.rgbaBytes[sourceIndex + 2].toInt() and 0xFF) / 255f

                pixels[targetIndex] = red
                pixels[planeSize + targetIndex] = green
                pixels[(planeSize * 2) + targetIndex] = blue
            }
        }

        return pixels
    }

    private fun updateBackground(frame: FloatArray) {
        if (!backgroundInitialized) {
            frame.copyInto(backgroundFrame)
            backgroundInitialized = true
            return
        }

        for (index in frame.indices) {
            backgroundFrame[index] += BACKGROUND_ALPHA * (frame[index] - backgroundFrame[index])
        }
    }

    private fun parseLatestHeatmap(heatmaps: Array<Array<FloatArray>>): RawBallCandidate? {
        val latestChannel = heatmaps.lastOrNull() ?: return null
        var selectedPeak = -1f
        var selectedPeakX = 0
        var selectedPeakY = 0

        for (y in latestChannel.indices) {
            val row = latestChannel[y]
            for (x in row.indices) {
                val score = row[x]
                if (score > selectedPeak) {
                    selectedPeak = score
                    selectedPeakX = x
                    selectedPeakY = y
                }
            }
        }

        if (selectedPeak < MIN_HEATMAP_CONFIDENCE) {
            return null
        }

        val threshold = max(selectedPeak * 0.55f, HEATMAP_FLOOR)
        var totalWeight = 0f
        var weightedX = 0f
        var weightedY = 0f
        var areaCount = 0

        val minY = max(0, selectedPeakY - SEARCH_RADIUS)
        val maxY = min(MODEL_HEIGHT - 1, selectedPeakY + SEARCH_RADIUS)
        val minX = max(0, selectedPeakX - SEARCH_RADIUS)
        val maxX = min(MODEL_WIDTH - 1, selectedPeakX + SEARCH_RADIUS)

        for (y in minY..maxY) {
            val row = latestChannel[y]
            for (x in minX..maxX) {
                val score = row[x]
                if (score < threshold) {
                    continue
                }

                totalWeight += score
                weightedX += x * score
                weightedY += y * score
                areaCount += 1
            }
        }

        if (totalWeight <= 0f) {
            return null
        }

        val centerX = weightedX / totalWeight
        val centerY = weightedY / totalWeight
        val normalizedRadius = (sqrt(areaCount.toFloat() / Math.PI.toFloat()) / MODEL_WIDTH).coerceIn(0.01f, 0.08f)

        return RawBallCandidate(
            centerX = centerX / MODEL_WIDTH,
            centerY = centerY / MODEL_HEIGHT,
            radius = normalizedRadius,
            score = selectedPeak.coerceIn(0f, 1f)
        )
    }

    private companion object {
        const val TAG = "TrackNetOnnxDetector"
        const val MODEL_PATH = "models/TrackNetV3-final.onnx"
        const val INPUT_NAME = "input"
        const val SEQUENCE_LENGTH = 8
        const val CHANNELS_PER_FRAME = 3
        const val MODEL_WIDTH = 512
        const val MODEL_HEIGHT = 288
        const val SEARCH_RADIUS = 18
        const val MIN_HEATMAP_CONFIDENCE = 0.18f
        const val HEATMAP_FLOOR = 0.10f
        const val BACKGROUND_ALPHA = 0.03f
    }
}
