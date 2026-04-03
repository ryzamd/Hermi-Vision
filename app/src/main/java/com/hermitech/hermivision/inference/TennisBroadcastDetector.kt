package com.hermitech.hermivision.inference

import android.content.Context

class TennisBroadcastDetector(
    context: Context
) : BallDetector {
    private val colorDetector = ColorBallDetector()
    private val trackNetDetector = TrackNetOnnxDetector(
        context = context,
        fallbackDetector = NullBallDetector()
    )

    override fun detect(frameData: FrameData): RawBallCandidate? {
        val colorCandidate = colorDetector.detect(frameData)
        if (colorCandidate != null && colorCandidate.score >= COLOR_CONFIDENCE_THRESHOLD) {
            return colorCandidate
        }

        val onnxCandidate = trackNetDetector.detect(frameData)
        if (onnxCandidate != null && onnxCandidate.score >= ONNX_CONFIDENCE_THRESHOLD) {
            return onnxCandidate
        }

        return colorCandidate ?: onnxCandidate
    }

    override fun close() {
        colorDetector.close()
        trackNetDetector.close()
    }

    private companion object {
        const val COLOR_CONFIDENCE_THRESHOLD = 0.16f
        const val ONNX_CONFIDENCE_THRESHOLD = 0.12f
    }
}
