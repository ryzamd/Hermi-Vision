package com.hermitech.hermivision.shared.domain.usecase

import com.hermitech.hermivision.shared.domain.model.AIConfig

class BuildBenchmarkResultsTextUseCase {
    operator fun invoke(config: AIConfig): String {
        val lines = mutableListOf<String>()
        lines.add("Device: ${config.deviceSummary}")
        lines.add("Best: ${config.tfliteDelegate.name}")
        if (config.nnapiAvgMs > 0) lines.add("NPU: ${config.nnapiAvgMs}ms/frame")
        if (config.gpuAvgMs > 0) lines.add("GPU: ${config.gpuAvgMs}ms/frame")
        if (config.cpuAvgMs > 0) lines.add("CPU: ${config.cpuAvgMs}ms/frame")
        return lines.joinToString("\n")
    }
}
