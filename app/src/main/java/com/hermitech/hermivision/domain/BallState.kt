package com.hermitech.hermivision.domain

data class NormalizedPoint(
    val x: Float,
    val y: Float
)

enum class TrackingStatus {
    TRACKING,
    LOST,
    REACQUIRED
}

data class BallState(
    val position: NormalizedPoint? = null,
    val radius: Float = 0f,
    val confidence: Float = 0f,
    val trail: List<NormalizedPoint> = emptyList(),
    val status: TrackingStatus = TrackingStatus.LOST,
    val fps: Float = 0f,
    val detectorSource: String = "none"
)
