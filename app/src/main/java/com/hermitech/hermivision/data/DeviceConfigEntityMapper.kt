package com.hermitech.hermivision.data

import com.hermitech.hermivision.data.model.DeviceConfigEntity
import com.hermitech.hermivision.shared.domain.model.AIConfig
import com.hermitech.hermivision.shared.domain.model.DelegateType
import com.hermitech.hermivision.shared.domain.model.DeviceTier

object DeviceConfigEntityMapper {
    fun toAIConfig(entity: DeviceConfigEntity): AIConfig = AIConfig(
        tier = DeviceTier.valueOf(entity.tier),
        tfliteDelegate = DelegateType.valueOf(entity.tfliteDelegate),
        yoloModelName = entity.yoloModelName,
        numThreads = entity.numThreads,
        deviceSummary = entity.deviceSummary,
        nnapiAvgMs = entity.nnapiAvgMs,
        gpuAvgMs = entity.gpuAvgMs,
        cpuAvgMs = entity.cpuAvgMs,
    )

    fun fromAIConfig(config: AIConfig): DeviceConfigEntity = DeviceConfigEntity(
        tier = config.tier.name,
        tfliteDelegate = config.tfliteDelegate.name,
        yoloModelName = config.yoloModelName,
        numThreads = config.numThreads,
        deviceSummary = config.deviceSummary,
        nnapiAvgMs = config.nnapiAvgMs,
        gpuAvgMs = config.gpuAvgMs,
        cpuAvgMs = config.cpuAvgMs,
    )
}
