package com.hermitech.hermivision.shared

import com.hermitech.hermivision.shared.domain.model.YoloLetterbox
import com.hermitech.hermivision.shared.domain.usecase.ParseYoloBallFrameUseCase
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ParseYoloBallFrameUseCaseTest {
    private val useCase = ParseYoloBallFrameUseCase(
        confThreshold = 0.25f,
        iouThreshold = 0.45f,
        numAnchors = 4,
        maxNmsCandidates = 4,
    )

    @Test
    fun returnsInvisibleFrameWhenNoCandidatePassesThreshold() {
        val output = Array(5) { FloatArray(4) }
        output[4][0] = 0.2f

        val frame = useCase(
            frameId = 7,
            output = output,
            letterbox = YoloLetterbox(scale = 1f, padX = 0, padY = 0),
        )

        assertFalse(frame.isVisible)
        assertEquals(null, frame.x)
        assertEquals(null, frame.y)
    }

    @Test
    fun keepsHighestScoringDetectionAfterNmsAndMapsCoordinatesBack() {
        val output = Array(5) { FloatArray(4) }

        output[0][0] = 100f
        output[1][0] = 120f
        output[2][0] = 20f
        output[3][0] = 20f
        output[4][0] = 0.9f

        output[0][1] = 102f
        output[1][1] = 122f
        output[2][1] = 20f
        output[3][1] = 20f
        output[4][1] = 0.8f

        val frame = useCase(
            frameId = 3,
            output = output,
            letterbox = YoloLetterbox(scale = 2f, padX = 10, padY = 20),
        )

        assertTrue(frame.isVisible)
        assertEquals(45f, frame.x)
        assertEquals(50f, frame.y)
    }
}
