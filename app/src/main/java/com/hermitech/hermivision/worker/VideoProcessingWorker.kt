package com.hermitech.hermivision.worker

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.hermitech.hermivision.data.AppDatabase
import com.hermitech.hermivision.domain.inference.TFLiteSessionManager
import com.hermitech.hermivision.domain.inference.YoloBallInferencer
import com.hermitech.hermivision.data.model.BallFrame
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import org.opencv.android.OpenCVLoader

/**
 * Background worker for ball detection pipeline.
 *
 * Simplified pipeline (v2 — YOLO26):
 *  1. Load AIConfig from Room DB (benchmark already done on first launch)
 *  2. Decode video → frame channel
 *  3. YOLO26 TFLite → ball positions (List<BallFrame>)
 *  4. Store results → UI shows numerical stats
 *
 * Court detection and bounce detection are temporarily disabled.
 */
class VideoProcessingWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {

    companion object {
        private const val TAG = "VideoProcessingWorker"
        const val KEY_VIDEO_URI = "video_uri"
        const val KEY_TOTAL_FRAMES = "total_frames"
        const val KEY_VISIBLE_FRAMES = "visible_frames"
        const val KEY_DURATION_MS = "duration_ms"
        const val KEY_STAGE = "stage"
        const val KEY_PROGRESS = "progress"
        const val KEY_CURRENT_FRAME = "current_frame"
        const val STAGE_DECODING = "Decoding video..."
        const val STAGE_BALL_TRACKING = "Detecting ball (YOLO26)..."
        const val STAGE_COMPLETE = "Complete"
    }

    /**
     * In-memory result holder.
     * WorkManager output Data is limited to 10KB, so store full results here.
     */
    object ResultHolder {
        var ballFrames: List<BallFrame> = emptyList()
        var totalFrames: Int = 0
        var visibleFrames: Int = 0
        var avgConfidence: Float = 0f
        var inferenceTimeMs: Long = 0L
        var durationMs: Long = 0L

        fun clear() {
            ballFrames = emptyList()
            totalFrames = 0
            visibleFrames = 0
            avgConfidence = 0f
            inferenceTimeMs = 0L
            durationMs = 0L
        }
    }

    override suspend fun doWork(): Result {
        val startTime = System.currentTimeMillis()
        ResultHolder.clear()

        // Initialize OpenCV
        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV initialization failed!")
            return Result.failure(workDataOf("error" to "OpenCV initialization failed"))
        }
        Log.i(TAG, "OpenCV initialized successfully")

        try {
            // --- Load AIConfig from Room DB (benchmark was done on first launch) ---
            val db = AppDatabase.getInstance(applicationContext)
            val configEntity = db.deviceConfigDao().getConfig()
                ?: return Result.failure(workDataOf("error" to "Device not optimized. Please restart the app."))
            val config = configEntity.toAIConfig()
            Log.i(TAG, "AI Config: ${config.deviceSummary}")

            // --- Stage 1: Decode video metadata ---
            reportProgress(STAGE_DECODING, 0)

            val videoPath = inputData.getString(KEY_VIDEO_URI)
                ?: return Result.failure(workDataOf("error" to "No video path provided"))

            val videoFile = java.io.File(videoPath)
            if (!videoFile.exists()) {
                return Result.failure(workDataOf("error" to "Video file not found: $videoPath"))
            }

            val retriever = android.media.MediaMetadataRetriever()
            retriever.setDataSource(videoPath)

            val totalFrames = retriever.extractMetadata(
                android.media.MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT
            )?.toIntOrNull() ?: 0

            val videoWidth = retriever.extractMetadata(
                android.media.MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH
            )?.toIntOrNull() ?: 0

            val videoHeight = retriever.extractMetadata(
                android.media.MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT
            )?.toIntOrNull() ?: 0

            retriever.release()

            if (totalFrames == 0 || videoWidth == 0 || videoHeight == 0) {
                return Result.failure(workDataOf("error" to "Cannot read video metadata"))
            }

            Log.i(TAG, "Video: $totalFrames frames, ${videoWidth}x${videoHeight}")
            reportProgress(STAGE_DECODING, 100)

            // --- Stage 2: YOLO26 Ball Detection ---
            reportProgress(STAGE_BALL_TRACKING, 0)

            val tfliteManager = TFLiteSessionManager(applicationContext)
            val yoloBall = YoloBallInferencer(tfliteManager)
            yoloBall.loadModel(config)

            val inferStartTime = System.currentTimeMillis()

            val ballFrames = try {
                val channel = kotlinx.coroutines.channels.Channel<org.opencv.core.Mat>(capacity = 4)
                val hardwareDecoder = com.hermitech.hermivision.domain.decoder.HardwareVideoDecoder()

                kotlinx.coroutines.coroutineScope {
                    // Decode video frames into channel
                    launch(Dispatchers.Default) {
                        try {
                            hardwareDecoder.decodeToMatChannel(videoPath, channel)
                        } finally {
                            hardwareDecoder.clearBuffer()
                        }
                    }

                    // YOLO26 inference from channel
                    val inferDeferred = async(Dispatchers.Default) {
                        yoloBall.inferFromChannel(channel, videoWidth, videoHeight) { processed ->
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
                yoloBall.releaseModel()
                tfliteManager.close()
            }

            val inferenceTimeMs = System.currentTimeMillis() - inferStartTime

            // Compute stats
            val visibleFrames = ballFrames.count { it.isVisible }

            ResultHolder.ballFrames = ballFrames
            ResultHolder.totalFrames = ballFrames.size
            ResultHolder.visibleFrames = visibleFrames
            ResultHolder.inferenceTimeMs = inferenceTimeMs

            Log.i(TAG, "Ball tracking done: $visibleFrames/${ballFrames.size} visible, ${inferenceTimeMs}ms")
            reportProgress(STAGE_BALL_TRACKING, 100)

            // --- Done ---
            val durationMs = System.currentTimeMillis() - startTime
            ResultHolder.durationMs = durationMs

            Log.i(TAG, "Pipeline complete in ${durationMs / 1000}s")
            reportProgress(STAGE_COMPLETE, 100)

            return Result.success(
                workDataOf(
                    KEY_TOTAL_FRAMES to ballFrames.size,
                    KEY_VISIBLE_FRAMES to visibleFrames,
                    KEY_DURATION_MS to durationMs
                )
            )

        } catch (e: Exception) {
            Log.e(TAG, "Pipeline failed", e)
            return Result.failure(workDataOf("error" to (e.message ?: "Unknown error")))
        }
    }

    private suspend fun reportProgress(stage: String, percent: Int) {
        setProgress(workDataOf(
            KEY_STAGE to stage,
            KEY_PROGRESS to percent
        ))
    }
}