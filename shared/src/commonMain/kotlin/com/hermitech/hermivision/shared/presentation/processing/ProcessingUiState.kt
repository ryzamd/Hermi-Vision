package com.hermitech.hermivision.shared.presentation.processing

data class ProcessingUiState(
    val isProcessing: Boolean = false,
    val isComplete: Boolean = false,
    val stage: String = "",
    val progressPercent: Int = 0,
    val currentFrame: Int = -1,
    val totalFrames: Int = 0,
    val visibleFrames: Int = 0,
    val durationMs: Long = 0L,
    val error: String? = null,
)
