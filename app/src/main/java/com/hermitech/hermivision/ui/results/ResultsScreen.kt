package com.hermitech.hermivision.ui.results

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.hermitech.hermivision.shared.domain.model.BallFrame
import com.hermitech.hermivision.shared.domain.usecase.CalculateProcessingSummaryUseCase

private val DarkBg = Color(0xFF0F0F23)
private val CardBg = Color(0xFF1A1A2E)
private val AccentColor = Color(0xFFFF6B35)
private val GreenColor = Color(0xFF00C853)
private val TextPrimary = Color(0xFFE0E0E0)
private val TextSecondary = Color(0xFF9E9E9E)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultsScreen(
    ballFrames: List<BallFrame>,
    totalFrames: Int,
    visibleFrames: Int,
    inferenceTimeMs: Long,
    totalDurationMs: Long = 0L,
    onBackClick: () -> Unit = {}
) {
    val summary = remember(ballFrames, totalFrames, visibleFrames, inferenceTimeMs, totalDurationMs) {
        CalculateProcessingSummaryUseCase()(
            ballFrames = if (ballFrames.size == totalFrames) ballFrames else ballFrames.take(totalFrames),
            inferenceTimeMs = inferenceTimeMs,
            totalDurationMs = totalDurationMs,
        )
    }

    Scaffold(
        topBar = {
            CenterAlignedTopAppBar(
                title = {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "Analysis Results",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = TextPrimary
                        )
                        if (totalDurationMs > 0) {
                            Text(
                                text = "Processed in %.1fs".format(totalDurationMs / 1000.0),
                                style = MaterialTheme.typography.labelSmall,
                                color = TextSecondary
                            )
                        }
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back",
                            tint = TextPrimary
                        )
                    }
                },
                colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                    containerColor = DarkBg
                )
            )
        },
        containerColor = DarkBg
    ) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            contentPadding = PaddingValues(vertical = 16.dp)
        ) {
            // Stats Cards
            item {
                Text(
                    text = "📊 Ball Tracking Stats",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = AccentColor,
                    modifier = Modifier.padding(bottom = 4.dp)
                )
            }

            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    StatCard(
                        label = "Total Frames",
                        value = totalFrames.toString(),
                        modifier = Modifier.weight(1f)
                    )
                    StatCard(
                        label = "Ball Detected",
                        value = visibleFrames.toString(),
                        valueColor = GreenColor,
                        modifier = Modifier.weight(1f)
                    )
                }
            }

            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    StatCard(
                        label = "Detection Rate",
                        value = "%.1f%%".format(summary.detectionRate),
                        valueColor = if (summary.detectionRate > 70) GreenColor else AccentColor,
                        modifier = Modifier.weight(1f)
                    )
                    StatCard(
                        label = "Inference FPS",
                        value = "%.1f".format(summary.inferenceFps),
                        modifier = Modifier.weight(1f)
                    )
                }
            }

            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    StatCard(
                        label = "Inference Time",
                        value = "%.1fs".format(inferenceTimeMs / 1000.0),
                        modifier = Modifier.weight(1f)
                    )
                    StatCard(
                        label = "Total Time",
                        value = "%.1fs".format(totalDurationMs / 1000.0),
                        modifier = Modifier.weight(1f)
                    )
                }
            }

            // Frame Details
            item {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "📋 Frame Details",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = AccentColor,
                    modifier = Modifier.padding(bottom = 4.dp)
                )
            }

            // Header row
            item {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(CardBg, shape = MaterialTheme.shapes.small)
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("Frame", color = AccentColor, fontWeight = FontWeight.Bold, fontSize = 13.sp, modifier = Modifier.weight(1f))
                    Text("Visible", color = AccentColor, fontWeight = FontWeight.Bold, fontSize = 13.sp, modifier = Modifier.weight(1f), textAlign = TextAlign.Center)
                    Text("X", color = AccentColor, fontWeight = FontWeight.Bold, fontSize = 13.sp, modifier = Modifier.weight(1f), textAlign = TextAlign.Center)
                    Text("Y", color = AccentColor, fontWeight = FontWeight.Bold, fontSize = 13.sp, modifier = Modifier.weight(1f), textAlign = TextAlign.End)
                }
            }

            // Data rows
            itemsIndexed(ballFrames) { _, frame ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 4.dp),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "${frame.frameId}",
                        color = TextSecondary,
                        fontSize = 12.sp,
                        modifier = Modifier.weight(1f)
                    )
                    Text(
                        text = if (frame.isVisible) "✓" else "✗",
                        color = if (frame.isVisible) GreenColor else Color(0xFFFF5252),
                        fontSize = 12.sp,
                        modifier = Modifier.weight(1f),
                        textAlign = TextAlign.Center
                    )
                    Text(
                        text = frame.x?.let { "%.1f".format(it) } ?: "—",
                        color = TextPrimary,
                        fontSize = 12.sp,
                        modifier = Modifier.weight(1f),
                        textAlign = TextAlign.Center
                    )
                    Text(
                        text = frame.y?.let { "%.1f".format(it) } ?: "—",
                        color = TextPrimary,
                        fontSize = 12.sp,
                        modifier = Modifier.weight(1f),
                        textAlign = TextAlign.End
                    )
                }
                if (frame.frameId < ballFrames.lastOrNull()?.frameId ?: 0) {
                    HorizontalDivider(color = CardBg, thickness = 0.5.dp)
                }
            }
        }
    }
}

@Composable
private fun StatCard(
    label: String,
    value: String,
    valueColor: Color = TextPrimary,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = CardBg),
        shape = MaterialTheme.shapes.medium
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = value,
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = valueColor
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = label,
                fontSize = 12.sp,
                color = TextSecondary
            )
        }
    }
}
