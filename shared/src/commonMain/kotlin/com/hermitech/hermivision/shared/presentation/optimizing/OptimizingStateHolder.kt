package com.hermitech.hermivision.shared.presentation.optimizing

class OptimizingStateHolder(
    initialState: OptimizingUiState = OptimizingUiState(),
) {
    var state: OptimizingUiState = initialState
        private set

    fun startBenchmark() {
        if (state.isRunning) return
        state = OptimizingUiState(
            isRunning = true,
            stage = stageForProgress(0),
        )
    }

    fun onBenchmarkProgress(progress: Int) {
        state = state.copy(
            progress = progress,
            stage = stageForProgress(progress),
        )
    }

    fun onBenchmarkCompleted(
        deviceSummary: String,
        benchmarkResults: String,
    ) {
        state = OptimizingUiState(
            isRunning = false,
            isDone = true,
            progress = 100,
            stage = "Optimization complete!",
            deviceSummary = deviceSummary,
            benchmarkResults = benchmarkResults,
        )
    }

    fun onBenchmarkFailed(message: String?) {
        state = OptimizingUiState(
            isRunning = false,
            error = buildString {
                append("Optimization failed")
                if (!message.isNullOrBlank()) {
                    append(": ")
                    append(message)
                }
            },
        )
    }

    private fun stageForProgress(progress: Int): String = when {
        progress < 10 -> "Analyzing hardware..."
        progress < 35 -> "Testing NPU (NNAPI)..."
        progress < 65 -> "Testing GPU..."
        progress < 85 -> "Testing CPU..."
        else -> "Finalizing..."
    }
}
