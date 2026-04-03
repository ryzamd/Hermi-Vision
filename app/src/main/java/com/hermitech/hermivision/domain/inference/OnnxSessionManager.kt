package com.hermitech.hermivision.domain.inference

import android.app.ActivityManager
import android.content.Context
import android.util.Log
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream

/**
 * Centralised ONNX session factory.
 *
 * Responsibilities:
 *  - Copy model from assets → internal filesDir (one-time)
 *  - Check available RAM before loading (MemoryInfo guard)
 *  - Create OrtSession with mmap + optimal thread config
 */
class OnnxSessionManager(private val context: Context) {

    companion object {
        private const val TAG = "OnnxSessionManager"
        private const val MIN_FREE_RAM_BYTES = 150L * 1024 * 1024
        private const val INTRA_OP_THREADS = 2
        private const val INTER_OP_THREADS = 1
    }

    val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    /**
     * Build shared SessionOptions used by every model.
     *
     * Thread config (2/1) keeps context-switching low on mobile.
     * Graph-level optimisation is applied once at load time.
     */
    fun buildSessionOptions(useNnapi: Boolean = true): OrtSession.SessionOptions =
        OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(INTRA_OP_THREADS)
            setInterOpNumThreads(INTER_OP_THREADS)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

            if (useNnapi) {
                try {
                    addNnapi()
                    Log.i(TAG, "NNAPI execution provider enabled")
                } catch (e: Exception) {
                    Log.w(TAG, "NNAPI not available, using CPU: ${e.message}")
                }
            }
        }

    /**
     * Load a model by asset name.
     *
     * Flow:
     *  1. Copy `assets/models/<name>` → `filesDir/models/<name>` (idempotent).
     *  2. Guard: check free RAM > 150 MB.
     *  3. Create session via *file path* so ORT uses mmap (native memory,
     *     not Java heap). OS can page-out unused pages when RAM is tight.
     */
    fun loadSession(assetModelName: String): OrtSession {
        val modelFile = copyAssetToFilesDir(assetModelName)
        checkMemoryGuard()

        // Try NNAPI first, fall back to CPU if model has unsupported ops
        return try {
            val nnapiOptions = buildSessionOptions(useNnapi = true)
            try {
                val session = env.createSession(modelFile.absolutePath, nnapiOptions)
                Log.i(TAG, "Loaded $assetModelName with NNAPI")
                session
            } finally {
                nnapiOptions.close()
            }
        } catch (e: Exception) {
            Log.w(TAG, "NNAPI failed for $assetModelName, falling back to CPU: ${e.message}")
            val cpuOptions = buildSessionOptions(useNnapi = false)
            try {
                val session = env.createSession(modelFile.absolutePath, cpuOptions)
                Log.i(TAG, "Loaded $assetModelName with CPU")
                session
            } finally {
                cpuOptions.close()
            }
        }
    }

    /**
     * Copy model from APK assets into the writable filesDir.
     * Skips if the destination already exists with the correct size.
     */
    private fun copyAssetToFilesDir(assetName: String): File {
        val destDir = File(context.filesDir, "models")
        if (!destDir.exists()) destDir.mkdirs()

        val destFile = File(destDir, File(assetName).name)
        if (destFile.exists()) return destFile          // already copied

        context.assets.open("models/$assetName").use { input ->
            FileOutputStream(destFile).use { output ->
                input.copyTo(output)
            }
        }
        return destFile
    }

    /**
     * Abort early if available RAM is dangerously low.
     *
     * This prevents a hard crash (SIGKILL by LMK) that would give
     * zero feedback to the user.
     */
    private fun checkMemoryGuard() {
        val activityManager =
            context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)

        if (memInfo.availMem < MIN_FREE_RAM_BYTES) {
            throw InsufficientMemoryException(
                "Not enough RAM to load model. " +
                "Available: ${memInfo.availMem / (1024 * 1024)} MB, " +
                "required: ${MIN_FREE_RAM_BYTES / (1024 * 1024)} MB"
            )
        }
    }
}

/** Thrown when available memory is too low to safely load a model. */
class InsufficientMemoryException(message: String) : RuntimeException(message)
