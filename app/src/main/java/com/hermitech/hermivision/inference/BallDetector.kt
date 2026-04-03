package com.hermitech.hermivision.inference

import java.io.Closeable
import kotlin.math.max
import kotlin.math.min

data class RawBallCandidate(
    val centerX: Float,
    val centerY: Float,
    val radius: Float,
    val score: Float
)

interface BallDetector : Closeable {
    fun detect(frameData: FrameData): RawBallCandidate?

    override fun close() = Unit
}

class ColorBallDetector : BallDetector {
    override fun detect(frameData: FrameData): RawBallCandidate? {
        val step = max(4, min(frameData.width, frameData.height) / 96)

        var candidateCount = 0
        var weightedX = 0f
        var weightedY = 0f
        var totalWeight = 0f
        var minX = frameData.width
        var minY = frameData.height
        var maxX = 0
        var maxY = 0

        for (y in 0 until frameData.height step step) {
            for (x in 0 until frameData.width step step) {
                val pixelIndex = y * frameData.rowStride + x * frameData.pixelStride
                if (pixelIndex + 2 >= frameData.rgbaBytes.size) {
                    continue
                }

                val red = frameData.rgbaBytes[pixelIndex].toInt() and 0xFF
                val green = frameData.rgbaBytes[pixelIndex + 1].toInt() and 0xFF
                val blue = frameData.rgbaBytes[pixelIndex + 2].toInt() and 0xFF

                if (!isBallLikeColor(red, green, blue)) {
                    continue
                }

                val weight = green.toFloat() / 255f
                candidateCount += 1
                totalWeight += weight
                weightedX += x * weight
                weightedY += y * weight
                minX = min(minX, x)
                minY = min(minY, y)
                maxX = max(maxX, x)
                maxY = max(maxY, y)
            }
        }

        if (candidateCount < 12 || totalWeight <= 0f) {
            return null
        }

        val boxWidth = (maxX - minX).coerceAtLeast(step)
        val boxHeight = (maxY - minY).coerceAtLeast(step)
        val aspectRatio = boxWidth.toFloat() / boxHeight.toFloat()
        if (aspectRatio !in 0.55f..1.85f) {
            return null
        }

        val estimatedArea = candidateCount * step * step.toFloat()
        val fillRatio = estimatedArea / (boxWidth * boxHeight).toFloat()
        if (fillRatio < 0.12f) {
            return null
        }

        val centerX = (weightedX / totalWeight) / frameData.width
        val centerY = (weightedY / totalWeight) / frameData.height
        val radius = (max(boxWidth, boxHeight) / 2f) / min(frameData.width, frameData.height)
        val coverageScore = (candidateCount / 60f).coerceIn(0f, 1f)
        val confidence = (fillRatio * 0.6f + coverageScore * 0.4f).coerceIn(0f, 1f)

        return RawBallCandidate(
            centerX = centerX,
            centerY = centerY,
            radius = radius.coerceIn(0.01f, 0.18f),
            score = confidence
        )
    }

    private fun isBallLikeColor(red: Int, green: Int, blue: Int): Boolean {
        return green in 115..255 &&
            red in 70..245 &&
            blue in 0..170 &&
            green > blue * 1.45f &&
            green >= red * 0.9f &&
            red > blue
    }
}
