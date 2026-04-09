package com.hermitech.hermivision.domain.inference

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Manages TFLite model loading and Interpreter lifecycle.
 *
 * Features:
 *  - Memory-mapped model loading (zero-copy from assets)
 *  - Config-driven hardware acceleration from DeviceProfiler
 *  - Fallback chain: starts from preferred delegate, falls back gracefully
 *  - Thread count configuration
 */
class TFLiteSessionManager(private val context: Context) {

    companion object {
        private const val TAG = "TFLiteSessionManager"
        private const val NUM_THREADS = 4
    }

    private var gpuDelegate: GpuDelegate? = null
    private var nnapiDelegate: NnApiDelegate? = null

    /** Reports which hardware tier was actually used for the last loaded model. */
    var activeHardware: String = "none"
        private set

    /**
     * Load a .tflite model from assets/models/ and create an Interpreter.
     *
     * The [preferredDelegate] determines the starting tier. If it fails, the
     * method falls back through lower tiers automatically:
     *   NNAPI → GPU → CPU
     *   GPU → CPU
     *   CPU (no fallback)
     *
     * @param modelName File name in assets/models/
     * @param preferredDelegate Best delegate from DeviceProfiler benchmark
     */
    fun loadInterpreter(
        modelName: String,
        preferredDelegate: DelegateType = DelegateType.NNAPI
    ): Interpreter {
        val modelBuffer = loadModelFile(modelName)

        // Build fallback chain starting from the preferred delegate
        val delegateChain = when (preferredDelegate) {
            DelegateType.NNAPI -> listOf(DelegateType.NNAPI, DelegateType.GPU, DelegateType.CPU)
            DelegateType.GPU -> listOf(DelegateType.GPU, DelegateType.CPU)
            DelegateType.CPU -> listOf(DelegateType.CPU)
        }

        for (delegateType in delegateChain) {
            try {
                val interpreter = tryCreateInterpreter(modelBuffer, delegateType)
                Log.i(TAG, "Loaded $modelName with $activeHardware")
                return interpreter
            } catch (e: Throwable) {
                Log.w(TAG, "${delegateType.name} delegate failed for $modelName: ${e.message}")
            }
        }

        // Should never reach here (CPU always works), but just in case
        throw RuntimeException("All delegates failed for $modelName")
    }

    /**
     * Attempt to create an Interpreter with a specific delegate.
     * Cleans up delegate resources on failure.
     */
    private fun tryCreateInterpreter(
        modelBuffer: MappedByteBuffer,
        delegateType: DelegateType
    ): Interpreter {
        var nnapiDel: NnApiDelegate? = null
        var gpuDel: GpuDelegate? = null

        try {
            modelBuffer.rewind()
            val options = Interpreter.Options().setNumThreads(NUM_THREADS)

            when (delegateType) {
                DelegateType.NNAPI -> {
                    nnapiDel = NnApiDelegate()
                    options.addDelegate(nnapiDel)
                }
                DelegateType.GPU -> {
                    gpuDel = GpuDelegate()
                    options.addDelegate(gpuDel)
                }
                DelegateType.CPU -> { /* XNNPACK auto-enabled */ }
            }

            val interpreter = Interpreter(modelBuffer, options)

            // Success — store delegate references for lifecycle management
            nnapiDelegate = nnapiDel
            gpuDelegate = gpuDel
            activeHardware = when (delegateType) {
                DelegateType.NNAPI -> "NPU (NNAPI)"
                DelegateType.GPU -> "GPU"
                DelegateType.CPU -> "CPU ($NUM_THREADS threads)"
            }

            return interpreter
        } catch (e: Throwable) {
            // Clean up delegate on failure before re-throwing
            nnapiDel?.close()
            gpuDel?.close()
            throw e
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
     * Release all delegate resources.
     */
    fun close() {
        nnapiDelegate?.close()
        nnapiDelegate = null
        gpuDelegate?.close()
        gpuDelegate = null
    }
}
