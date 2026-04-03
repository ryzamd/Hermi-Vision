package com.hermitech.hermivision.tracking

import com.hermitech.hermivision.domain.BallState
import com.hermitech.hermivision.domain.DetectionResult
import com.hermitech.hermivision.domain.NormalizedPoint
import com.hermitech.hermivision.domain.TrackingStatus
import kotlin.math.max

class BallTracker(
    private val smoothingAlpha: Float = 0.28f,
    private val maxMissedFrames: Int = 6,
    private val trailSize: Int = 18
) {
    private var smoothedPoint: NormalizedPoint? = null
    private var smoothedRadius: Float = 0f
    private var lastConfidence: Float = 0f
    private var missedFrames: Int = maxMissedFrames + 1
    private val trail = ArrayDeque<NormalizedPoint>()

    @Synchronized
    fun update(detectionResult: DetectionResult?, fps: Float): BallState {
        if (detectionResult == null) {
            missedFrames += 1
            if (missedFrames > maxMissedFrames) {
                smoothedPoint = null
                smoothedRadius = 0f
                trail.clear()
                return BallState(
                    position = null,
                    radius = 0f,
                    confidence = 0f,
                    trail = emptyList(),
                    status = TrackingStatus.LOST,
                    fps = fps,
                    detectorSource = "none"
                )
            }

            return BallState(
                position = smoothedPoint,
                radius = smoothedRadius,
                confidence = lastConfidence * 0.8f,
                trail = trail.toList(),
                status = TrackingStatus.LOST,
                fps = fps,
                detectorSource = "hold"
            )
        }

        val previousPoint = smoothedPoint
        val nextPoint = if (previousPoint == null) {
            NormalizedPoint(detectionResult.x, detectionResult.y)
        } else {
            NormalizedPoint(
                x = previousPoint.x + smoothingAlpha * (detectionResult.x - previousPoint.x),
                y = previousPoint.y + smoothingAlpha * (detectionResult.y - previousPoint.y)
            )
        }

        smoothedPoint = nextPoint
        smoothedRadius = if (smoothedRadius == 0f) {
            detectionResult.radius
        } else {
            smoothedRadius + smoothingAlpha * (detectionResult.radius - smoothedRadius)
        }
        lastConfidence = detectionResult.confidence

        val status = if (missedFrames > 0) {
            TrackingStatus.REACQUIRED
        } else {
            TrackingStatus.TRACKING
        }
        missedFrames = 0

        trail.addLast(nextPoint)
        while (trail.size > trailSize) {
            trail.removeFirst()
        }

        return BallState(
            position = nextPoint,
            radius = max(smoothedRadius, 0.01f),
            confidence = detectionResult.confidence,
            trail = trail.toList(),
            status = status,
            fps = fps,
            detectorSource = "live"
        )
    }

    @Synchronized
    fun reset() {
        smoothedPoint = null
        smoothedRadius = 0f
        lastConfidence = 0f
        missedFrames = maxMissedFrames + 1
        trail.clear()
    }
}
