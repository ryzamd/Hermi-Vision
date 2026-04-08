package com.hermitech.hermivision.domain.inference

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Manages TFLite model loading and Interpreter lifecycle.
 *
 * Features:
 *  - Memory-mapped model loading (zero-copy from assets)
 *  - GPU Delegate with automatic CPU fallback
 *  - Thread count configuration
 */
class TFLiteSessionManager(private val context: Context) {

    companion object {
        private const val TAG = "TFLiteSessionManager"
        private const val NUM_THREADS = 4
    }

    private var gpuDelegate: GpuDelegate? = null

    /**
     * Load a .tflite model from assets/models/ and create an Interpreter.
     * Tries GPU delegate first, falls back to CPU if GPU is not available.
     */
    fun loadInterpreter(modelName: String): Interpreter {
        val modelBuffer = loadModelFile(modelName)

        // Tier 1: Try GPU delegate
        return try {
            val delegate = GpuDelegate()
            val options = Interpreter.Options()
                .addDelegate(delegate)
                .setNumThreads(NUM_THREADS)

            val interpreter = Interpreter(modelBuffer, options)
            gpuDelegate = delegate
            Log.i(TAG, "Loaded $modelName with GPU delegate")
            interpreter
        } catch (e: Throwable) {
            Log.w(TAG, "GPU delegate failed: ${e.message}")

            // Tier 2: Plain CPU fallback
            val options = Interpreter.Options()
                .setNumThreads(NUM_THREADS)

            val interpreter = Interpreter(modelBuffer, options)
            Log.i(TAG, "Loaded $modelName with CPU ($NUM_THREADS threads)")
            interpreter
        }
    }

    /**
     * Memory-map the model file from assets.
     * This avoids copying the entire model into RAM.
     */
    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val assetPath = "models/$modelName"
        val fd = context.assets.openFd(assetPath)
        val inputStream = FileInputStream(fd.fileDescriptor)
        val channel = inputStream.channel
        val startOffset = fd.startOffset
        val declaredLength = fd.declaredLength
        val buffer = channel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        inputStream.close()
        fd.close()
        return buffer
    }

    /**
     * Release GPU delegate resources.
     */
    fun close() {
        gpuDelegate?.close()
        gpuDelegate = null
    }
}
