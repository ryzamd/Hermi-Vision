package com.hermitech.hermivision.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse

class AppEnvironmentTest {
    @Test
    fun defaultBallDetectionConfigUsesBestModel() {
        val config = BallDetectionEntrypoint.defaultConfig()

        assertEquals("best", config.modelName)
        assertEquals(null, config.modelPathOverride)
        assertFalse(AppEnvironment.ballDetectionFeatureFlag())
    }
}
