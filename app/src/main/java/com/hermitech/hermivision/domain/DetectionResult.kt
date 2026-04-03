package com.hermitech.hermivision.domain

data class DetectionResult(
    val x: Float,
    val y: Float,
    val confidence: Float,
    val radius: Float
)
