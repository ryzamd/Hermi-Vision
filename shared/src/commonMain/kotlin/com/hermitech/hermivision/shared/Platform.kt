package com.hermitech.hermivision.shared

interface Platform {
    val name: String
}

expect fun getPlatform(): Platform
