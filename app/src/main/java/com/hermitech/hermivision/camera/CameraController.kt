package com.hermitech.hermivision.camera

import android.content.Context
import android.view.Surface
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.Closeable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class CameraController(
    private val context: Context
) : Closeable {

    private val mainExecutor = ContextCompat.getMainExecutor(context)
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
    private val isClosed = AtomicBoolean(false)
    private var imageAnalysis: ImageAnalysis? = null

    fun bind(
        previewView: PreviewView,
        lifecycleOwner: LifecycleOwner,
        analyzer: ImageAnalysis.Analyzer
    ) {
        val targetRotation = previewView.display?.rotation ?: Surface.ROTATION_0

        cameraProviderFuture.addListener(
            {
                if (isClosed.get()) {
                    return@addListener
                }

                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder()
                    .setTargetRotation(targetRotation)
                    .build()
                    .also { it.surfaceProvider = previewView.surfaceProvider }

                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setTargetRotation(targetRotation)
                    .build()
                    .also { imageAnalysis ->
                        imageAnalysis.setAnalyzer(cameraExecutor, analyzer)
                    }

                imageAnalysis = analysis
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis
                )
            },
            mainExecutor
        )
    }

    override fun close() {
        if (!isClosed.compareAndSet(false, true)) {
            return
        }

        imageAnalysis?.clearAnalyzer()
        if (!cameraExecutor.isShutdown) {
            cameraExecutor.shutdown()
        }

        if (cameraProviderFuture.isDone) {
            cameraProviderFuture.get().unbindAll()
        }
    }
}
