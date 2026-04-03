package com.hermitech.hermivision.domain

data class BounceEvent(
    val id: Long,
    val position: NormalizedPoint,
    val confidence: Float,
    val timestampMs: Long
)
