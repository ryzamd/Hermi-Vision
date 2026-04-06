package com.hermitech.hermivision.worker

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.hermitech.hermivision.domain.inference.BounceDetectorInferencer
import com.hermitech.hermivision.domain.inference.CourtDetectorInferencer
import com.hermitech.hermivision.domain.inference.OnnxSessionManager
import com.hermitech.hermivision.domain.inference.TrackNetInferencer
import com.hermitech.hermivision.domain.inference.CourtProjector
import com.hermitech.hermivision.data.model.BallFrame
import com.hermitech.hermivision.data.model.CourtMatrix
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import androidx.core.net.toUri

/**
 * Background worker that chains the full tennis analysis pipeline.
 *
 * Pipeline stages:
 *  1. Decode video → frame count + lazy frame provider
 *  2. TrackNet      → ball trajectory (List<BallFrame>)
 *  3. CourtDetector → homography matrices (List<CourtMatrix>)
 *  4. BounceDetector → bounce frame IDs (Set<Int>)
 *  5. CourtProjector → bounce positions on 2D court
 *
 * Progress stages are reported via setProgress() for UI updates.
 *
 * Input data keys:
 *  - KEY_VIDEO_URI: Content URI string of the video
 *
 * Output data keys:
 *  - KEY_TOTAL_FRAMES, KEY_BOUNCE_COUNT, KEY_DURATION_MS
 *
 * Note: Full results (ball frames, court matrices, bounces) are stored
 * in-memory via a companion object ResultHolder for simplicity.
 * Production app should use Room DB.
 */
class VideoProcessingWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {

    companion object {
        private const val TAG = "VideoProcessingWorker"
        const val KEY_VIDEO_URI = "video_uri"
        const val KEY_TOTAL_FRAMES = "total_frames"
        const val KEY_BOUNCE_COUNT = "bounce_count"
        const val KEY_DURATION_MS = "duration_ms"
        const val KEY_STAGE = "stage"
        const val KEY_PROGRESS = "progress"
        const val KEY_CURRENT_FRAME = "current_frame"
        const val STAGE_DECODING = "Decoding video..."
        const val STAGE_BALL_TRACKING = "Tracking ball..."
        const val STAGE_COURT_DETECTION = "Detecting court..."
        const val STAGE_BOUNCE_DETECTION = "Detecting bounces..."
        const val STAGE_PROJECTION = "Projecting results..."
        const val STAGE_COMPLETE = "Complete"
    }

    /**
     * In-memory result holder.
     *
     * WorkManager's output Data is limited to 10KB, so we store
     * full results here for the UI to read after completion.
     * In production, replace with Room DB.
     */
    object ResultHolder {
        var ballFrames: List<BallFrame> = emptyList()
        var courtMatrices: List<CourtMatrix> = emptyList()
        var bounceFrameIds: Set<Int> = emptySet()
        var bounceCourtPoints: List<Pair<Int, android.graphics.PointF>> = emptyList()
        var trajectoryCourtPoints: List<Pair<Int, android.graphics.PointF>> = emptyList()
        var durationMs: Long = 0L

        fun clear() {
            ballFrames = emptyList()
            courtMatrices = emptyList()
            bounceFrameIds = emptySet()
            bounceCourtPoints = emptyList()
            trajectoryCourtPoints = emptyList()
            durationMs = 0L
        }
    }

    override suspend fun doWork(): Result {
        val startTime = System.currentTimeMillis()

        ResultHolder.clear()

        // Initialize OpenCV native library
        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV initialization failed!")
            return Result.failure(workDataOf("error" to "OpenCV initialization failed"))
        }
        Log.i(TAG, "OpenCV initialized successfully")

        try {
            // --- Stage 0: Decode video metadata ---
            reportProgress(STAGE_DECODING, 0)

            val retriever = MediaMetadataRetriever()
            val videoPath = inputData.getString(KEY_VIDEO_URI)
                ?: return Result.failure(workDataOf("error" to "No video path provided"))

            val videoFile = java.io.File(videoPath)
            if (!videoFile.exists()) {
                return Result.failure(workDataOf("error" to "Video file not found: $videoPath"))
            }
            retriever.setDataSource(videoPath)

            val totalFrames = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT
            )?.toIntOrNull() ?: 0

            val videoWidth = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH
            )?.toIntOrNull() ?: 0

            val videoHeight = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT
            )?.toIntOrNull() ?: 0

            if (totalFrames == 0 || videoWidth == 0 || videoHeight == 0) {
                retriever.release()
                return Result.failure(workDataOf("error" to "Cannot read video metadata"))
            }

            Log.i(TAG, "Video: $totalFrames frames, ${videoWidth}x${videoHeight}")
            reportProgress(STAGE_DECODING, 100)

            val sessionManager = OnnxSessionManager(applicationContext)

            reportProgress(STAGE_BALL_TRACKING, 0)

            val trackNet = TrackNetInferencer(sessionManager)
            trackNet.loadModel()

            val ballFrames = try {
                val channel = kotlinx.coroutines.channels.Channel<org.opencv.core.Mat>(capacity = TrackNetInferencer.SEQ_LEN)
                val hardwareDecoder = com.hermitech.hermivision.domain.decoder.HardwareVideoDecoder()

                kotlinx.coroutines.coroutineScope {
                    // Decode Video bằng MediaCodec và nhúng khung hình trực tiếp thành C++ Mat
                    launch(kotlinx.coroutines.Dispatchers.Default) {
                        try {
                            hardwareDecoder.decodeToMatChannel(videoPath, channel)
                        } finally {
                            hardwareDecoder.clearBuffer()
                        }
                    }

                    // Luồng NPU liên tục kéo Data từ Channel thả vào ONNX
                    val inferDeferred = async(kotlinx.coroutines.Dispatchers.Default) {
                        trackNet.inferFromChannel(channel, videoWidth, videoHeight) { processed ->
                            if (totalFrames > 0 && (processed % 10 == 0 || processed == totalFrames)) {
                                val pct = (processed * 100) / totalFrames
                                setProgressAsync(workDataOf(
                                    KEY_STAGE to STAGE_BALL_TRACKING,
                                    KEY_PROGRESS to pct,
                                    KEY_CURRENT_FRAME to processed
                                ))
                            }
                        }
                    }

                    inferDeferred.await()
                }
            } finally {
                trackNet.releaseModel()
            }

            ResultHolder.ballFrames = ballFrames
            Log.i(TAG, "Ball tracking done: ${ballFrames.count { it.isVisible }}/${ballFrames.size} visible")
            reportProgress(STAGE_BALL_TRACKING, 100)

            // Court detection
            reportProgress(STAGE_COURT_DETECTION, 0)

            val courtDetector = CourtDetectorInferencer(sessionManager)
            courtDetector.loadModel()

            val courtMatrices = try {
                val channel = kotlinx.coroutines.channels.Channel<org.opencv.core.Mat>(capacity = 9)
                val hardwareDecoder = com.hermitech.hermivision.domain.decoder.HardwareVideoDecoder()

                kotlinx.coroutines.coroutineScope {
                    launch(Dispatchers.Default) {
                        try {
                            hardwareDecoder.decodeToMatChannel(videoPath, channel)
                        } finally {
                            hardwareDecoder.clearBuffer()
                        }
                    }

                    val inferDeferred = async(Dispatchers.Default) {
                        courtDetector.inferFromChannel(channel, useRefine = true) { processed ->
                            if (totalFrames > 0 && (processed % 10 == 0 || processed == totalFrames)) {
                                val pct = (processed * 100) / totalFrames
                                setProgressAsync(workDataOf(
                                    KEY_STAGE to STAGE_COURT_DETECTION,
                                    KEY_PROGRESS to pct,
                                    KEY_CURRENT_FRAME to processed
                                ))
                            }
                        }
                    }
                    
                    inferDeferred.await()
                }
            } finally {
                courtDetector.releaseModel()
            }

            ResultHolder.courtMatrices = courtMatrices
            val courtDetected = courtMatrices.count { it.homographyInv != null }
            Log.i(TAG, "Court detection done: $courtDetected/${courtMatrices.size} frames with court")
            reportProgress(STAGE_COURT_DETECTION, 100)

            // Release retriever
            retriever.release()

            // Bounce detection
            reportProgress(STAGE_BOUNCE_DETECTION, 0)

            val bounceDetector = BounceDetectorInferencer(sessionManager)
            val bounceFrameIds = try {
                bounceDetector.detectBounces(ballFrames)
            } finally {
                bounceDetector.releaseSession()
            }

            ResultHolder.bounceFrameIds = bounceFrameIds
            Log.i(TAG, "Bounce detection done: ${bounceFrameIds.size} bounces detected")
            reportProgress(STAGE_BOUNCE_DETECTION, 100)

            // Court projection
            reportProgress(STAGE_PROJECTION, 0)

            val projector = CourtProjector()
            val bounceCourtPoints = projector.projectBounces(ballFrames, courtMatrices, bounceFrameIds)
            val trajectoryCourtPoints = projector.projectTrajectory(ballFrames, courtMatrices)

            ResultHolder.bounceCourtPoints = bounceCourtPoints
            ResultHolder.trajectoryCourtPoints = trajectoryCourtPoints
            Log.i(TAG, "Projection done: ${bounceCourtPoints.size} bounces projected, ${trajectoryCourtPoints.size} trajectory points")
            reportProgress(STAGE_PROJECTION, 100)

            // Done
            val durationMs = System.currentTimeMillis() - startTime
            ResultHolder.durationMs = durationMs

            Log.i(TAG, "Pipeline complete in ${durationMs / 1000}s")
            reportProgress(STAGE_COMPLETE, 100)

            return Result.success(
                workDataOf(
                    KEY_TOTAL_FRAMES to totalFrames,
                    KEY_BOUNCE_COUNT to bounceFrameIds.size,
                    KEY_DURATION_MS to durationMs
                )
            )

        } catch (e: Exception) {
            Log.e(TAG, "Pipeline failed", e)
            return Result.failure(workDataOf("error" to (e.message ?: "Unknown error")))
        }
    }

    private fun loadFrame(retriever: MediaMetadataRetriever, frameIndex: Int): Bitmap {
        return retriever.getFrameAtIndex(frameIndex)
            ?: throw RuntimeException("Failed to decode frame $frameIndex")
    }

    private suspend fun reportProgress(stage: String, percent: Int) {
        setProgress(workDataOf(
            KEY_STAGE to stage,
            KEY_PROGRESS to percent
        ))
    }
}