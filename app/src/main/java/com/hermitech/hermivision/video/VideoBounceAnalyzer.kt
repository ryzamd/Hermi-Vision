package com.hermitech.hermivision.video

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import com.hermitech.hermivision.domain.BallState
import com.hermitech.hermivision.domain.BounceEvent
import com.hermitech.hermivision.inference.BallDetector
import com.hermitech.hermivision.inference.Postprocessor
import com.hermitech.hermivision.inference.Preprocessor
import com.hermitech.hermivision.tracking.BallTracker
import com.hermitech.hermivision.tracking.BounceDetector
import kotlinx.coroutines.ensureActive
import kotlin.coroutines.coroutineContext

data class VideoAnalysisResult(
    val lastBallState: BallState,
    val bounceEvents: List<BounceEvent>
)

class VideoBounceAnalyzer(
    private val context: Context,
    private val preprocessor: Preprocessor,
    private val detector: BallDetector,
    private val postprocessor: Postprocessor,
    private val tracker: BallTracker,
    private val bounceDetector: BounceDetector
) {
    suspend fun analyze(
        videoUri: Uri,
        onProgress: (processedFrames: Int, totalFramesEstimate: Int, latestBallState: BallState) -> Unit
    ): VideoAnalysisResult {
        tracker.reset()
        bounceDetector.reset()

        val bounceEvents = mutableListOf<BounceEvent>()
        var lastBallState = BallState()

        MediaMetadataRetriever().use { retriever ->
            retriever.setDataSource(context, videoUri)

            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull()
                ?.coerceAtLeast(1L)
                ?: 1L
            val intervalMs = FRAME_INTERVAL_MS
            val totalFramesEstimate = (durationMs / intervalMs).toInt().coerceAtLeast(1)

            var processedFrames = 0
            var timeMs = 0L
            while (timeMs <= durationMs) {
                coroutineContext.ensureActive()
                val bitmap = retriever.getFrameAtTime(timeMs * 1_000L, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                if (bitmap != null) {
                    val frameData = preprocessor.preprocess(bitmap)
                    bitmap.recycle()

                    val rawCandidate = detector.detect(frameData)
                    val detection = postprocessor.postprocess(rawCandidate)
                    lastBallState = tracker.update(detection, fps = ANALYSIS_FPS)
                    val bounceEvent = bounceDetector.update(lastBallState)
                    if (bounceEvent != null) {
                        bounceEvents += bounceEvent
                    }
                    onProgress(processedFrames, totalFramesEstimate, lastBallState)
                }

                processedFrames += 1
                timeMs += intervalMs
            }
        }

        return VideoAnalysisResult(
            lastBallState = lastBallState,
            bounceEvents = bounceEvents
        )
    }

    private companion object {
        const val FRAME_INTERVAL_MS = 66L
        const val ANALYSIS_FPS = 1000f / FRAME_INTERVAL_MS
    }
}
