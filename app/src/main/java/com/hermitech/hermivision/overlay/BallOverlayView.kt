package com.hermitech.hermivision.overlay

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.hermitech.hermivision.domain.BallState
import com.hermitech.hermivision.domain.BounceEvent
import com.hermitech.hermivision.domain.TrackingStatus

@Composable
fun BallOverlayView(
    ballState: BallState,
    detectorLabel: String,
    bounceEvents: List<BounceEvent>,
    modifier: Modifier = Modifier
) {
    val accentColor = when (ballState.status) {
        TrackingStatus.TRACKING -> Color(0xFFB8FF3D)
        TrackingStatus.REACQUIRED -> Color(0xFFFFD54F)
        TrackingStatus.LOST -> Color(0xFFFF6B6B)
    }

    Box(modifier = modifier) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            with(TrailRenderer) {
                drawTrail(trail = ballState.trail, color = accentColor)
            }

            ballState.position?.let { point ->
                val center = Offset(point.x * size.width, point.y * size.height)
                val radius = (ballState.radius * size.minDimension).coerceAtLeast(size.minDimension * 0.02f)

                drawCircle(
                    color = accentColor.copy(alpha = 0.22f),
                    radius = radius * 1.75f,
                    center = center
                )
                drawCircle(
                    color = accentColor,
                    radius = radius,
                    center = center
                )
            }

            bounceEvents.forEachIndexed { index, event ->
                val center = Offset(event.position.x * size.width, event.position.y * size.height)
                val alpha = 0.25f + (index + 1) / bounceEvents.size.toFloat() * 0.55f
                drawCircle(
                    color = Color(0xFFFF7043).copy(alpha = alpha),
                    radius = size.minDimension * 0.02f,
                    center = center
                )
                drawCircle(
                    color = Color.White.copy(alpha = alpha),
                    radius = size.minDimension * 0.008f,
                    center = center
                )
            }
        }

        Column(
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(16.dp)
                .background(Color(0xAA08111D), RoundedCornerShape(18.dp))
                .padding(horizontal = 14.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Text(
                text = detectorLabel,
                color = Color.White,
                style = MaterialTheme.typography.labelLarge
            )
            Text(
                text = "Status: ${ballState.status.name}",
                color = accentColor,
                fontWeight = FontWeight.SemiBold,
                style = MaterialTheme.typography.bodyMedium
            )
            Text(
                text = "FPS: ${"%.1f".format(ballState.fps)}",
                color = Color.White,
                style = MaterialTheme.typography.bodySmall
            )
            Text(
                text = "Confidence: ${"%.2f".format(ballState.confidence)}",
                color = Color.White,
                style = MaterialTheme.typography.bodySmall
            )
            Text(
                text = "Bounces: ${bounceEvents.size}",
                color = Color(0xFFFFD180),
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}
