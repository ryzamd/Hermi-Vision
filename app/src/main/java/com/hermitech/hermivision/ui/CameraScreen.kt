package com.hermitech.hermivision.ui

import android.net.Uri
import android.widget.VideoView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.hermitech.hermivision.R

@Composable
fun VideoAnalysisRoute(
    onPickVideo: () -> Unit,
    modifier: Modifier = Modifier,
    viewModel: CameraViewModel = viewModel()
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    Surface(modifier = modifier.fillMaxSize(), color = Color(0xFF02060D)) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text(
                    text = stringResource(R.string.video_input_title),
                    style = MaterialTheme.typography.headlineSmall,
                    color = Color.White
                )
                Text(
                    text = stringResource(R.string.video_input_body),
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color.White.copy(alpha = 0.82f)
                )
            }

            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = onPickVideo) {
                    Text(text = stringResource(R.string.pick_video))
                }
                Button(
                    onClick = viewModel::analyzeSelectedVideo,
                    enabled = uiState.selectedVideoUri != null && !uiState.isAnalyzingVideo
                ) {
                    Text(text = stringResource(R.string.analyze_video))
                }
            }

            Text(
                text = uiState.selectedVideoUri?.toString() ?: stringResource(R.string.no_video_selected),
                style = MaterialTheme.typography.bodySmall,
                color = Color.White.copy(alpha = 0.72f)
            )

            VideoPreview(
                videoUri = uiState.selectedVideoUri,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(240.dp)
            )

            if (uiState.isAnalyzingVideo) {
                LinearProgressIndicator(
                    progress = { uiState.analysisProgress },
                    modifier = Modifier.fillMaxWidth()
                )
            }

            Text(
                text = uiState.analysisStatus,
                style = MaterialTheme.typography.bodyMedium,
                color = Color(0xFFFFE0B2)
            )

            BounceLogPanel(
                uiState = uiState,
                onClearBounceLog = viewModel::clearBounceLog
            )
        }
    }
}

@Composable
private fun VideoPreview(
    videoUri: Uri?,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier.background(Color(0xFF06101B)),
        contentAlignment = Alignment.Center
    ) {
        if (videoUri == null) {
            Text(
                text = stringResource(R.string.no_video_selected),
                color = Color.White.copy(alpha = 0.82f)
            )
        } else {
            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { context ->
                    VideoView(context).apply {
                        setVideoURI(videoUri)
                        seekTo(100)
                    }
                },
                update = { view ->
                    view.setVideoURI(videoUri)
                    view.seekTo(100)
                }
            )
        }
    }
}

@Composable
private fun BounceLogPanel(
    uiState: CameraUiState,
    onClearBounceLog: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .background(Color(0xCC08111D))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(
                    text = stringResource(R.string.bounce_log_title),
                    style = MaterialTheme.typography.titleSmall,
                    color = Color.White
                )
                Text(
                    text = uiState.bounceSessionPath,
                    style = MaterialTheme.typography.bodySmall,
                    color = Color.White.copy(alpha = 0.72f)
                )
            }

            Button(onClick = onClearBounceLog) {
                Text(text = stringResource(R.string.clear_bounce_log))
            }
        }

        if (uiState.bounceEvents.isEmpty()) {
            Text(
                text = stringResource(R.string.no_bounce_detected),
                style = MaterialTheme.typography.bodyMedium,
                color = Color.White.copy(alpha = 0.82f)
            )
        } else {
            uiState.bounceEvents.takeLast(8).reversed().forEach { event ->
                Text(
                    text = "Bounce #${event.id}: x=${"%.3f".format(event.position.x)}  y=${"%.3f".format(event.position.y)}  conf=${"%.2f".format(event.confidence)}",
                    style = MaterialTheme.typography.bodySmall,
                    color = Color(0xFFFFE0B2)
                )
            }
        }
    }
}
