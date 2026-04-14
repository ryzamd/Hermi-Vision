package com.hermitech.hermivision.data

import com.hermitech.hermivision.shared.data.DeviceConfigRepository
import com.hermitech.hermivision.shared.domain.model.AIConfig

class RoomDeviceConfigRepository(
    private val db: AppDatabase,
) : DeviceConfigRepository {
    override suspend fun getConfig(): AIConfig? =
        db.deviceConfigDao().getConfig()?.let(DeviceConfigEntityMapper::toAIConfig)

    override suspend fun saveConfig(config: AIConfig) {
        db.deviceConfigDao().insertConfig(DeviceConfigEntityMapper.fromAIConfig(config))
    }
}
