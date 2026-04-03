package com.hermitech.hermivision.domain.decoder

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext

data class VideoMetadata(
    val width: Int,
    val height: Int,
    val fps: Float
)

class VideoDecoder(private val context: Context) {

    fun extractFramesAsFlow(videoUri: Uri): Flow<Bitmap> = flow {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, videoUri)
            val frameCountStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)
            val frameCount = frameCountStr?.toIntOrNull() ?: 0
            
            if (frameCount > 0) {
                for (i in 0 until frameCount) {
                    val frameBitmap = retriever.getFrameAtIndex(i)
                    if (frameBitmap != null) {
                        emit(frameBitmap)
                    } else {
                        // Handle missing frames
                    }
                }
            } else {
                // Fallback if metadata is missing
                val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                val durationMs = durationStr?.toLongOrNull() ?: 0L
                val fps = 30L
                val intervalUs = 1_000_000L / fps
                val durationUs = durationMs * 1000L
                
                var timeUs = 0L
                while (timeUs < durationUs) {
                    val frameBitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                    if (frameBitmap != null) {
                        emit(frameBitmap)
                    }
                    timeUs += intervalUs
                }
            }
        } finally {
            retriever.release()
        }
    }.flowOn(Dispatchers.IO)
    
    suspend fun getVideoMetadata(videoUri: Uri): VideoMetadata = withContext(Dispatchers.IO) {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, videoUri)
            val width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
            val height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
            val frameCountStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)
            val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            val frameCount = frameCountStr?.toIntOrNull() ?: 0
            val durationMs = durationStr?.toLongOrNull() ?: 0L
            val fps = if (durationMs > 0) (frameCount * 1000f / durationMs) else 30f

            VideoMetadata(width, height, fps)
        } finally {
            retriever.release()
        }
    }
}