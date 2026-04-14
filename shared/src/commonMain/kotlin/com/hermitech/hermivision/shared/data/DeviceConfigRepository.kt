package com.hermitech.hermivision.shared.data

import com.hermitech.hermivision.shared.domain.model.AIConfig

interface DeviceConfigRepository {
    suspend fun getConfig(): AIConfig?
    suspend fun saveConfig(config: AIConfig)
}
