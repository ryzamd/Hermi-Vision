package com.hermitech.hermivision.inference

import android.graphics.Bitmap
import androidx.camera.core.ImageProxy

data class FrameData(
    val width: Int,
    val height: Int,
    val rgbaBytes: ByteArray,
    val rowStride: Int,
    val pixelStride: Int
)

class Preprocessor {
    fun preprocess(imageProxy: ImageProxy): FrameData {
        val plane = imageProxy.planes.first()
        val rgbaBuffer = plane.buffer
        val rgbaBytes = ByteArray(rgbaBuffer.remaining())
        rgbaBuffer.get(rgbaBytes)

        return FrameData(
            width = imageProxy.width,
            height = imageProxy.height,
            rgbaBytes = rgbaBytes,
            rowStride = plane.rowStride,
            pixelStride = plane.pixelStride
        )
    }

    fun preprocess(bitmap: Bitmap): FrameData {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val rgbaBytes = ByteArray(width * height * 4)
        var offset = 0
        pixels.forEach { pixel ->
            rgbaBytes[offset] = ((pixel shr 16) and 0xFF).toByte()
            rgbaBytes[offset + 1] = ((pixel shr 8) and 0xFF).toByte()
            rgbaBytes[offset + 2] = (pixel and 0xFF).toByte()
            rgbaBytes[offset + 3] = ((pixel ushr 24) and 0xFF).toByte()
            offset += 4
        }

        return FrameData(
            width = width,
            height = height,
            rgbaBytes = rgbaBytes,
            rowStride = width * 4,
            pixelStride = 4
        )
    }
}
