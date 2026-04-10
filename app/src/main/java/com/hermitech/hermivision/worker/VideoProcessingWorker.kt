package com.hermitech.hermivision.worker

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.hermitech.hermivision.data.AppDatabase
import com.hermitech.hermivision.domain.inference.NativePipeline
import com.hermitech.hermivision.data.model.BallFrame
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import org.opencv.android.OpenCVLoader
import java.io.File

/**
 * Background worker for ball detection pipeline.
 *
 * Pipeline v3 — C++ Native:
 *  1. Load AIConfig from Room DB (benchmark already done on first launch)
 *  2. Extract .tflite model to cache dir (one-time)
 *  3. Init C++ pipeline (NativePipeline → JNI → HermiVisionPipeline)
 *  4. Decode video → Mat channel → C++ inference per frame (zero-copy)
 *  5. Store results → UI shows numerical stats
 *
 * All heavy computation (letterbox, normalize, TFLite invoke, NMS)
 * runs in native C++ — zero GC pressure, pre-allocated buffers.
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
        const val STAGE_BALL_TRACKING = "Detecting ball (YOLO26 C++)..."
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

        // Initialize OpenCV (needed for HardwareVideoDecoder Mat creation)
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

            // --- Stage 2: C++ Native Ball Detection ---
            reportProgress(STAGE_BALL_TRACKING, 0)

            // Extract .tflite model from assets to cache dir (TFLite C API needs file path)
            val modelPath = extractModelToCache(applicationContext, config.yoloModelName)
            Log.i(TAG, "Model extracted: $modelPath")

            // Init C++ pipeline via JNI
            pipeline = NativePipeline()
            val ok = pipeline.init(
                delegateType = config.tfliteDelegate.ordinal,  // 0=NNAPI, 1=GPU, 2=CPU
                numThreads = config.numThreads,
                modelPath = modelPath
            )
            if (!ok) {
                return Result.failure(workDataOf("error" to "C++ pipeline initialization failed"))
            }
            Log.i(TAG, "C++ pipeline ready — delegate: ${pipeline.getActiveDelegate()}")

            val inferStartTime = System.currentTimeMillis()

            // Producer–Consumer: decode → channel → C++ inference
            val ballFrames = run {
                val channel = kotlinx.coroutines.channels.Channel<org.opencv.core.Mat>(capacity = 8)
                val hardwareDecoder = com.hermitech.hermivision.domain.decoder.HardwareVideoDecoder()

                kotlinx.coroutines.coroutineScope {
                    // Producer: decode video → Mat channel
                    launch(Dispatchers.Default) {
                        try {
                            hardwareDecoder.decodeToMatChannel(videoPath, channel)
                        } finally {
                            hardwareDecoder.clearBuffer()
                        }
                    }

                    // Consumer: C++ inference per frame (zero-copy via Mat address)
                    val inferDeferred = async(Dispatchers.Default) {
                        val results = mutableListOf<BallFrame>()
                        var frameId = 0

                        for (mat in channel) {
                            // Pass native Mat pointer → C++ (ZERO pixel copy)
                            val result = pipeline.processFrame(
                                matAddr = mat.nativeObjAddr,
                                frameId = frameId,
                                origWidth = videoWidth,
                                origHeight = videoHeight
                            )
                            // result = [ballVisible(0/1), ballX, ballY, ballScore]

                            results.add(BallFrame(
                                frameId = frameId,
                                isVisible = result[0] > 0.5f,
                                x = result[1],
                                y = result[2]
                            ))

                            mat.release()
                            frameId++

                            // Report progress every 10 frames
                            if (totalFrames > 0 && (frameId % 10 == 0 || frameId == totalFrames)) {
                                val pct = (frameId * 100) / totalFrames
                                setProgressAsync(workDataOf(
                                    KEY_STAGE to STAGE_BALL_TRACKING,
                                    KEY_PROGRESS to pct,
                                    KEY_CURRENT_FRAME to frameId
                                ))
                            }
                        }
                        results
                    }

                    inferDeferred.await()
                }
            }

            val inferenceTimeMs = System.currentTimeMillis() - inferStartTime

            // Compute stats
            val visibleFrames = ballFrames.count { it.isVisible }

            ResultHolder.ballFrames = ballFrames
            ResultHolder.totalFrames = ballFrames.size
            ResultHolder.visibleFrames = visibleFrames
            ResultHolder.inferenceTimeMs = inferenceTimeMs

            Log.i(TAG, "Ball tracking done: $visibleFrames/${ballFrames.size} visible, ${inferenceTimeMs}ms")
            Log.i(TAG, "  Delegate used: ${pipeline.getActiveDelegate()}")
            val fps = if (inferenceTimeMs > 0) ballFrames.size * 1000.0 / inferenceTimeMs else 0.0
            Log.i(TAG, "  Effective FPS: ${"%.1f".format(fps)} (${ballFrames.size} frames / ${inferenceTimeMs}ms)")
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
            // Always release native resources
            pipeline?.release()
        }
    }

    /**
     * Extract a model file from assets to cache directory.
     * TFLite C API requires a file path (cannot read Android assets directly).
     * Only copies if file doesn't already exist in cache.
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