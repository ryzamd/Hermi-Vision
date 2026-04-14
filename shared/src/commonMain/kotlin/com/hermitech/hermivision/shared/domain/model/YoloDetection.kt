package com.hermitech.hermivision.shared.domain.model

data class YoloDetection(
    val cx: Float,
    val cy: Float,
    val w: Float,
    val h: Float,
    val score: Float,
)
