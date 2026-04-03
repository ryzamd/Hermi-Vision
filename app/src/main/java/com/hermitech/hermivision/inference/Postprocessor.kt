package com.hermitech.hermivision.inference

import com.hermitech.hermivision.domain.DetectionResult

class Postprocessor(
    private val minConfidence: Float = 0.22f
) {
    fun postprocess(candidate: RawBallCandidate?): DetectionResult? {
        if (candidate == null || candidate.score < minConfidence) {
            return null
        }

        return DetectionResult(
            x = candidate.centerX.coerceIn(0f, 1f),
            y = candidate.centerY.coerceIn(0f, 1f),
            confidence = candidate.score.coerceIn(0f, 1f),
            radius = candidate.radius.coerceIn(0.01f, 0.2f)
        )
    }
}
