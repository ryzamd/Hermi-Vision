package com.hermitech.hermivision.camera

import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.hermitech.hermivision.domain.BallState
import com.hermitech.hermivision.inference.BallDetector
import com.hermitech.hermivision.inference.Postprocessor
import com.hermitech.hermivision.inference.Preprocessor
import com.hermitech.hermivision.tracking.BallTracker
import java.io.Closeable
import java.util.concurrent.atomic.AtomicBoolean

class FrameAnalyzer(
    private val preprocessor: Preprocessor,
    private val detector: BallDetector,
    private val postprocessor: Postprocessor,
    private val tracker: BallTracker,
    private val onBallStateUpdated: (BallState) -> Unit
) : ImageAnalysis.Analyzer, Closeable {

    private val isProcessing = AtomicBoolean(false)
    private var frameCounter = 0
    private var fpsWindowStartedAt = SystemClock.elapsedRealtime()
    private var latestFps = 0f

    override fun analyze(imageProxy: ImageProxy) {
        if (!isProcessing.compareAndSet(false, true)) {
            imageProxy.close()
            return
        }

        try {
            val frameData = preprocessor.preprocess(imageProxy)
            val rawCandidate = detector.detect(frameData)
            val detection = postprocessor.postprocess(rawCandidate)
            val updatedState = tracker.update(detection, fps = updateFps())
            onBallStateUpdated(updatedState)
        } catch (exception: Exception) {
            Log.e(TAG, "Failed to analyze camera frame", exception)
            onBallStateUpdated(tracker.update(detectionResult = null, fps = latestFps))
        } finally {
            imageProxy.close()
            isProcessing.set(false)
        }
    }

    override fun close() {
        detector.close()
        tracker.reset()
    }

    private fun updateFps(): Float {
        frameCounter += 1
        val now = SystemClock.elapsedRealtime()
        val elapsed = now - fpsWindowStartedAt
        if (elapsed >= 1_000L) {
            latestFps = frameCounter * 1_000f / elapsed
            frameCounter = 0
            fpsWindowStartedAt = now
        }
        return latestFps
    }

    private companion object {
        const val TAG = "FrameAnalyzer"
    }
}
