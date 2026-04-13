package com.hermitech.hermivision.worker

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.hermitech.hermivision.data.AppDatabase
import com.hermitech.hermivision.domain.inference.NativePipeline
import com.hermitech.hermivision.domain.inference.DelegateType
import com.hermitech.hermivision.data.model.BallFrame
import com.hermitech.hermivision.data.model.CourtResult
import com.hermitech.hermivision.domain.decoder.HardwareVideoDecoder
import org.opencv.android.OpenCVLoader
import java.io.File

/**
 * Background worker for ball detection pipeline.
 *
 * Pipeline v4 — FramePool Native:
 *  1. Load AIConfig from Room DB (benchmark already done on first launch)
 *  2. Extract .tflite model to cache dir (one-time)
 *  3. Init C++ pipeline (NativePipeline → JNI → HermiVisionPipeline)
 *  4. Decode video → YUV planes → submitYuvFrame() → C++ FramePool
 *  5. processLatestFrame() → YOLO acquires from pool → results
 *  6. Store results → UI shows numerical stats
 *
 * Zero-allocation path: decoder YUV → C++ pool → YOLO preprocess → TFLite
 * No Channel<Mat>, no clone(), no Kotlin-side RGB conversion.
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
        const val STAGE_BALL_TRACKING = "Detecting ball + court (YOLO26 + MobileNet C++)..."
        const val STAGE_COMPLETE = "Complete"

        // Court model name (MobileNetV3-Small, not device-dependent)
        private const val COURT_MODEL_NAME = "Court-Keypoint-FP32.tflite"
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
        var courtResult: CourtResult = CourtResult.EMPTY

        fun clear() {
            ballFrames = emptyList()
            totalFrames = 0
            visibleFrames = 0
            avgConfidence = 0f
            inferenceTimeMs = 0L
            durationMs = 0L
            courtResult = CourtResult.EMPTY
        }
    }

    override suspend fun doWork(): Result {
        val startTime = System.currentTimeMillis()
        ResultHolder.clear()

        // Initialize OpenCV (still needed for some utility functions)
        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV initialization failed!")
            return Result.failure(workDataOf("error" to "OpenCV initialization failed"))
        }
        Log.i(TAG, "OpenCV initialized successfully")

        var pipeline: NativePipeline? = null

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

            val videoFile = File(videoPath)
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

            // --- Stage 2: C++ Native Ball Detection (FramePool path) ---
            reportProgress(STAGE_BALL_TRACKING, 0)

            val modelPath = extractModelToCache(applicationContext, config.yoloModelName)
            Log.i(TAG, "Ball model extracted: $modelPath")

            // Extract court model to cache
            val courtModelPath = extractModelToCache(applicationContext, COURT_MODEL_NAME)
            Log.i(TAG, "Court model extracted: $courtModelPath")

            // Init C++ pipeline via JNI (ball + court models)
            pipeline = NativePipeline()
            val ok = pipeline.init(
                delegateType = config.tfliteDelegate.ordinal,
                numThreads = config.numThreads,
                ballModelPath = modelPath,
                courtModelPath = courtModelPath,
                courtDelegateType = DelegateType.CPU.ordinal  // Court runs on CPU (GPU busy with YOLO)
            )
            if (!ok) {
                return Result.failure(workDataOf("error" to "C++ pipeline initialization failed"))
            }
            Log.i(TAG, "C++ pipeline ready — ball delegate: ${pipeline.getActiveDelegate()}")

            val inferStartTime = System.currentTimeMillis()

            // Sequential FramePool flow (video analysis — process EVERY frame):
            //   decode frame → submitYuvFrame → processLatestFrame → next frame
            // This ensures each frame is processed exactly once (no skipping, no duplicates)
            var lastCourtResult = CourtResult.EMPTY

            val ballFrames = run {
                val hardwareDecoder = HardwareVideoDecoder()
                val results = mutableListOf<BallFrame>()

                // Sequential: decoder submits each frame, then we process it immediately
                hardwareDecoder.decodeToFramePool(videoPath, pipeline) { frameId ->
                    // Process the frame that was just submitted to the pool
                    val result = pipeline.processLatestFrame()

                    results.add(BallFrame(
                        frameId = frameId,
                        isVisible = result[NativePipeline.IDX_BALL_VISIBLE] > 0.5f,
                        x = result[NativePipeline.IDX_BALL_X],
                        y = result[NativePipeline.IDX_BALL_Y]
                    ))

                    // Capture court keypoints (updated every ~30 frames by background thread)
                    if (result.size >= NativePipeline.RESULT_SIZE &&
                        result[NativePipeline.IDX_COURT_VALID] > 0.5f) {
                        val kps = mutableListOf<Pair<Float, Float>>()
                        for (i in 0 until 14) {
                            val x = result[NativePipeline.IDX_COURT_KP_START + i * 2]
                            val y = result[NativePipeline.IDX_COURT_KP_START + i * 2 + 1]
                            kps.add(x to y)
                        }
                        lastCourtResult = CourtResult(valid = true, keypoints = kps)
                    }

                    // Report progress every 10 frames
                    if (totalFrames > 0 && (frameId % 10 == 0 || frameId + 1 == totalFrames)) {
                        val pct = ((frameId + 1) * 100) / totalFrames
                        setProgressAsync(workDataOf(
                            KEY_STAGE to STAGE_BALL_TRACKING,
                            KEY_PROGRESS to pct,
                            KEY_CURRENT_FRAME to (frameId + 1)
                        ))
                    }
                }

                results
            }

            val inferenceTimeMs = System.currentTimeMillis() - inferStartTime

            // Compute stats
            val visibleFrames = ballFrames.count { it.isVisible }

            ResultHolder.ballFrames = ballFrames
            ResultHolder.totalFrames = ballFrames.size
            ResultHolder.visibleFrames = visibleFrames
            ResultHolder.inferenceTimeMs = inferenceTimeMs
            ResultHolder.courtResult = lastCourtResult

            Log.i(TAG, "Ball tracking done: $visibleFrames/${ballFrames.size} visible, ${inferenceTimeMs}ms")
            Log.i(TAG, "  Delegate used: ${pipeline.getActiveDelegate()}")
            val fps = if (inferenceTimeMs > 0) ballFrames.size * 1000.0 / inferenceTimeMs else 0.0
            Log.i(TAG, "  Effective FPS: ${"%.1f".format(fps)} (${ballFrames.size} frames / ${inferenceTimeMs}ms)")
            Log.i(TAG, "  Court detected: ${lastCourtResult.valid}")
            if (lastCourtResult.valid) {
                Log.i(TAG, "  Court keypoints: ${lastCourtResult.keypoints.size} points")
            }
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
        } finally {
            pipeline?.release()
        }
    }

    /**
     * Extract a model file from assets to cache directory.
     */
    private fun extractModelToCache(context: Context, modelName: String): String {
        val cacheFile = File(context.cacheDir, modelName)
        if (!cacheFile.exists()) {
            Log.i(TAG, "Extracting model to cache: $modelName")
            context.assets.open("models/$modelName").use { input ->
                cacheFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return cacheFile.absolutePath
    }

    private suspend fun reportProgress(stage: String, percent: Int) {
        setProgress(workDataOf(
            KEY_STAGE to stage,
            KEY_PROGRESS to percent
        ))
    }
}