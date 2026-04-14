package com.hermitech.hermivision.shared.domain.model

data class ProcessingSummary(
    val totalFrames: Int,
    val visibleFrames: Int,
    val inferenceTimeMs: Long = 0L,
    val totalDurationMs: Long = 0L,
) {
    val detectionRate: Float
        get() = if (totalFrames > 0) visibleFrames * 100f / totalFrames else 0f

    val inferenceFps: Float
        get() = if (inferenceTimeMs > 0) totalFrames * 1000f / inferenceTimeMs else 0f
}
