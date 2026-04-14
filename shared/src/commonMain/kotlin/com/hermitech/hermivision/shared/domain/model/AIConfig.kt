package com.hermitech.hermivision.shared.domain.model

enum class DeviceTier {
    HIGH,
    MEDIUM,
    LOW,
}

enum class DelegateType {
    NNAPI,
    GPU,
    CPU,
}

data class AIConfig(
    val tier: DeviceTier,
    val tfliteDelegate: DelegateType,
    val yoloModelName: String,
    val numThreads: Int,
    val deviceSummary: String,
    val nnapiAvgMs: Long = -1,
    val gpuAvgMs: Long = -1,
    val cpuAvgMs: Long = -1,
)
