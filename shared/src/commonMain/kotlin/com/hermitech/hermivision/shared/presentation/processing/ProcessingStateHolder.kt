package com.hermitech.hermivision.shared.presentation.processing

class ProcessingStateHolder(
    initialState: ProcessingUiState = ProcessingUiState(),
) {
    var state: ProcessingUiState = initialState
        private set

    fun onCopyStarted() {
        state = ProcessingUiState(
            isProcessing = true,
            stage = "Copying video...",
        )
    }

    fun onCopyFailed() {
        state = ProcessingUiState(
            error = "Failed to copy video file",
        )
    }

    fun onProcessingQueued() {
        state = ProcessingUiState(
            isProcessing = true,
            stage = "Starting...",
        )
    }

    fun onProgress(
        stage: String,
        progressPercent: Int,
        currentFrame: Int,
    ) {
        state = ProcessingUiState(
            isProcessing = true,
            stage = stage,
            progressPercent = progressPercent,
            currentFrame = currentFrame,
        )
    }

    fun onCompleted(
        totalFrames: Int,
        visibleFrames: Int,
        durationMs: Long,
    ) {
        state = ProcessingUiState(
            isProcessing = false,
            isComplete = true,
            stage = "Complete",
            progressPercent = 100,
            totalFrames = totalFrames,
            visibleFrames = visibleFrames,
            durationMs = durationMs,
        )
    }

    fun onFailed(message: String?) {
        state = ProcessingUiState(
            isProcessing = false,
            isComplete = false,
            error = message ?: "Unknown error",
        )
    }

    fun onCancelled() {
        state = ProcessingUiState(
            isProcessing = false,
            error = "Processing cancelled",
        )
    }
}
