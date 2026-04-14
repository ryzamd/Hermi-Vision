package com.hermitech.hermivision.data.model

import androidx.room.Entity
import androidx.room.PrimaryKey

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
)
