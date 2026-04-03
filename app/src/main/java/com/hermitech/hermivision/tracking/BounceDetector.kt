package com.hermitech.hermivision.tracking

import android.os.SystemClock
import com.hermitech.hermivision.domain.BallState
import com.hermitech.hermivision.domain.BounceEvent
import com.hermitech.hermivision.domain.NormalizedPoint
import com.hermitech.hermivision.domain.TrackingStatus
import kotlin.math.abs

class BounceDetector(
    private val minBounceY: Float = 0.38f,
    private val minVerticalTravel: Float = 0.018f,
    private val minConfidence: Float = 0.12f,
    private val cooldownFrames: Int = 10
) {
    private data class TimedPoint(
        val point: NormalizedPoint,
        val confidence: Float,
        val frameIndex: Long,
        val timestampMs: Long
    )

    private val points = ArrayDeque<TimedPoint>()
    private var frameIndex: Long = 0
    private var lastBounceFrame: Long = -cooldownFrames.toLong()
    private var nextEventId: Long = 1

    @Synchronized
    fun update(ballState: BallState): BounceEvent? {
        frameIndex += 1
        val point = ballState.position
        if (point == null || ballState.status == TrackingStatus.LOST || ballState.confidence < minConfidence) {
            if (points.size > 2) {
                points.removeFirst()
            }
            return null
        }

        points.addLast(
            TimedPoint(
                point = point,
                confidence = ballState.confidence,
                frameIndex = frameIndex,
                timestampMs = SystemClock.elapsedRealtime()
            )
        )
        while (points.size > WINDOW_SIZE) {
            points.removeFirst()
        }

        if (points.size < WINDOW_SIZE) {
            return null
        }

        val values = points.toList()
        val candidate = values[WINDOW_CENTER]
        val before = values[WINDOW_CENTER - 1]
        val after = values[WINDOW_CENTER + 1]
        val before2 = values[WINDOW_CENTER - 2]
        val after2 = values[WINDOW_CENTER + 2]

        val isCooldownActive = candidate.frameIndex - lastBounceFrame < cooldownFrames
        if (isCooldownActive) {
            return null
        }

        val descendingIntoBounce = candidate.point.y > before.point.y &&
            before.point.y >= before2.point.y &&
            candidate.point.y - before2.point.y >= minVerticalTravel
        val ascendingAfterBounce = candidate.point.y > after.point.y &&
            after.point.y >= after2.point.y &&
            candidate.point.y - after2.point.y >= minVerticalTravel
        val isNearCourt = candidate.point.y >= minBounceY
        val lateralJump = maxOf(
            abs(candidate.point.x - before.point.x),
            abs(after.point.x - candidate.point.x)
        )

        if (!descendingIntoBounce || !ascendingAfterBounce || !isNearCourt || lateralJump > 0.22f) {
            return null
        }

        lastBounceFrame = candidate.frameIndex
        return BounceEvent(
            id = nextEventId++,
            position = candidate.point,
            confidence = values.map { it.confidence }.average().toFloat(),
            timestampMs = candidate.timestampMs
        )
    }

    @Synchronized
    fun reset() {
        points.clear()
        frameIndex = 0
        lastBounceFrame = -cooldownFrames.toLong()
        nextEventId = 1
    }

    private companion object {
        const val WINDOW_SIZE = 5
        const val WINDOW_CENTER = 2
    }
}
