package com.hermitech.hermivision.shared.presentation.optimizing

data class OptimizingUiState(
    val isRunning: Boolean = false,
    val isDone: Boolean = false,
    val progress: Int = 0,
    val stage: String = "Preparing...",
    val deviceSummary: String = "",
    val benchmarkResults: String = "",
    val error: String? = null,
)
