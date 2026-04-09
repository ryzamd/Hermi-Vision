package com.hermitech.hermivision.data.model

import androidx.room.Entity
import androidx.room.PrimaryKey
import com.hermitech.hermivision.domain.inference.AIConfig
import com.hermitech.hermivision.domain.inference.DelegateType
import com.hermitech.hermivision.domain.inference.DeviceTier

@Entity(tableName = "device_config")
data class DeviceConfigEntity(
    @PrimaryKey
    val id: Int = 1,
    val tier: String,
    val tfliteDelegate: String,
    val yoloModelName: String,
    val numThreads: Int,
    val deviceSummary: String,
    val nnapiAvgMs: Long,
    val gpuAvgMs: Long,
    val cpuAvgMs: Long
) {
    fun toAIConfig(): AIConfig = AIConfig(
        tier = DeviceTier.valueOf(tier),
        tfliteDelegate = DelegateType.valueOf(tfliteDelegate),
        yoloModelName = yoloModelName,
        numThreads = numThreads,
        deviceSummary = deviceSummary,
        nnapiAvgMs = nnapiAvgMs,
        gpuAvgMs = gpuAvgMs,
        cpuAvgMs = cpuAvgMs
    )

    companion object {
        fun fromAIConfig(config: AIConfig): DeviceConfigEntity = DeviceConfigEntity(
            tier = config.tier.name,
            tfliteDelegate = config.tfliteDelegate.name,
            yoloModelName = config.yoloModelName,
            numThreads = config.numThreads,
            deviceSummary = config.deviceSummary,
            nnapiAvgMs = config.nnapiAvgMs,
            gpuAvgMs = config.gpuAvgMs,
            cpuAvgMs = config.cpuAvgMs
        )
    }
}
