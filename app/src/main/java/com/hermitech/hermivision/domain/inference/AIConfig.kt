package com.hermitech.hermivision.domain.inference

enum class DeviceTier {
    /** Real NPU available and fast: Snapdragon 8xx, Dimensity 9000+, Exynos 2200+ */
    HIGH,
    /** Strong GPU, limited/no NPU: Snapdragon 7xx, Helio G99, Mali-G78+ */
    MEDIUM,
    /** CPU-focused: Helio G85, Snapdragon 4xx, low RAM devices */
    LOW
}

enum class DelegateType {
    /** NNAPI — routes to NPU/DSP on supported chipsets */
    NNAPI,
    /** GPU delegate — OpenGL/OpenCL compute shaders */
    GPU,
    /** CPU with XNNPACK auto-optimization */
    CPU
}

/**
 * AI inference configuration determined by device profiling.
 *
 * Created once during first-launch benchmark, cached in persistent storage,
 * and used by all session managers to configure optimal delegates.
 *
 * @property tier Device capability tier (HIGH/MEDIUM/LOW)
 * @property tfliteDelegate Best TFLite delegate for this device
 * @property yoloModelName TFLite model file to load (e.g. "YoloBall-nano-FP16.tflite")
 * @property numThreads CPU thread count for inference
 * @property deviceSummary Human-readable summary (e.g. "Samsung A05 (mt6769v) → LOW (GPU)")
 * @property nnapiAvgMs NNAPI benchmark avg ms per frame (-1 if unavailable)
 * @property gpuAvgMs GPU benchmark avg ms per frame (-1 if unavailable)
 * @property cpuAvgMs CPU benchmark avg ms per frame (-1 if unavailable)
 */
data class AIConfig(
    val tier: DeviceTier,
    val tfliteDelegate: DelegateType,
    val yoloModelName: String,
    val numThreads: Int,
    val deviceSummary: String,
    val nnapiAvgMs: Long = -1,
    val gpuAvgMs: Long = -1,
    val cpuAvgMs: Long = -1
)