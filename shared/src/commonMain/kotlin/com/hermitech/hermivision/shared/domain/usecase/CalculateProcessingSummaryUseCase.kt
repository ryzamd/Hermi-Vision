package com.hermitech.hermivision.shared.domain.usecase

import com.hermitech.hermivision.shared.domain.model.BallFrame
import com.hermitech.hermivision.shared.domain.model.ProcessingSummary

class CalculateProcessingSummaryUseCase {
    operator fun invoke(
        ballFrames: List<BallFrame>,
        inferenceTimeMs: Long = 0L,
        totalDurationMs: Long = 0L,
    ): ProcessingSummary {
        val visibleFrames = ballFrames.count { it.isVisible }
        return ProcessingSummary(
            totalFrames = ballFrames.size,
            visibleFrames = visibleFrames,
            inferenceTimeMs = inferenceTimeMs,
            totalDurationMs = totalDurationMs,
        )
    }
}
