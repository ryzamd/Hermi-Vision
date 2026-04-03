package com.hermitech.hermivision.ui.results

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.hermitech.hermivision.data.model.BallFrame

private val BounceRowColor = Color(0x30FF6B35)
private val BounceIndicatorColor = Color(0xFFFF6B35)
private val VisibleColor = Color(0xFF4CAF50)
private val InvisibleColor = Color(0xFF9E9E9E)
private val HeaderBg = Color(0xFF1A1A2E)
private val HeaderText = Color(0xFFE0E0E0)

@Composable
fun BallTrackTable(ballFrames: List<BallFrame>, bounceFrameIds: Set<Int>, modifier: Modifier = Modifier) {
    Column(modifier = modifier.fillMaxSize()) {
        SummaryRow(
            totalFrames = ballFrames.size,
            visibleFrames = ballFrames.count { it.isVisible },
            bounceCount = bounceFrameIds.size
        )

        Spacer(modifier = Modifier.height(8.dp))

        TableHeader()

        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(bottom = 16.dp)
        ) {
            items(
                items = ballFrames,
                key = { it.frameId }
            ) { frame ->
                val isBounce = frame.frameId in bounceFrameIds
                TableRow(frame = frame, isBounce = isBounce)
            }
        }
    }
}

@Composable
private fun SummaryRow(totalFrames: Int, visibleFrames: Int, bounceCount: Int) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        SummaryChip(
            label = "Frames",
            value = "$totalFrames",
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.weight(1f)
        )
        SummaryChip(
            label = "Visible",
            value = "$visibleFrames",
            color = VisibleColor,
            modifier = Modifier.weight(1f)
        )
        SummaryChip(
            label = "Bounces",
            value = "$bounceCount",
            color = BounceIndicatorColor,
            modifier = Modifier.weight(1f)
        )
    }
}

@Composable
private fun SummaryChip(label: String, value: String, color: Color, modifier: Modifier = Modifier) {
    Surface(
        modifier = modifier
            .clip(RoundedCornerShape(12.dp)),
        color = color.copy(alpha = 0.12f),
        tonalElevation = 1.dp
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = value,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = color
            )
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = color.copy(alpha = 0.7f)
            )
        }
    }
}

@Composable
private fun TableHeader() {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(HeaderBg)
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        HeaderCell("Frame", Modifier.weight(1.2f))
        HeaderCell("Vis", Modifier.weight(0.6f))
        HeaderCell("X", Modifier.weight(1f))
        HeaderCell("Y", Modifier.weight(1f))
        HeaderCell("", Modifier.weight(0.6f))
    }
}

@Composable
private fun HeaderCell(text: String, modifier: Modifier = Modifier) {
    Text(
        text = text,
        modifier = modifier,
        color = HeaderText,
        fontSize = 12.sp,
        fontWeight = FontWeight.SemiBold,
        textAlign = TextAlign.Center,
        letterSpacing = 0.5.sp
    )
}

@Composable
private fun TableRow(frame: BallFrame, isBounce: Boolean) {
    val bgColor = if (isBounce) BounceRowColor else Color.Transparent

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(bgColor)
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = "${frame.frameId}",
            modifier = Modifier.weight(1.2f),
            textAlign = TextAlign.Center,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = if (isBounce) FontWeight.Bold else FontWeight.Normal
        )

        Text(
            text = if (frame.isVisible) "✓" else "✗",
            modifier = Modifier.weight(0.6f),
            textAlign = TextAlign.Center,
            color = if (frame.isVisible) VisibleColor else InvisibleColor,
            fontWeight = FontWeight.Bold
        )

        Text(
            text = frame.x?.let { "%.1f".format(it) } ?: "—",
            modifier = Modifier.weight(1f),
            textAlign = TextAlign.Center,
            style = MaterialTheme.typography.bodyMedium,
            color = if (frame.isVisible)
                MaterialTheme.colorScheme.onSurface
            else
                MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f)
        )

        Text(
            text = frame.y?.let { "%.1f".format(it) } ?: "—",
            modifier = Modifier.weight(1f),
            textAlign = TextAlign.Center,
            style = MaterialTheme.typography.bodyMedium,
            color = if (frame.isVisible)
                MaterialTheme.colorScheme.onSurface
            else
                MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f)
        )

        Text(
            text = if (isBounce) "★" else "",
            modifier = Modifier.weight(0.6f),
            textAlign = TextAlign.Center,
            color = BounceIndicatorColor,
            fontWeight = FontWeight.Bold,
            fontSize = 14.sp
        )
    }

    HorizontalDivider(
        color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.3f),
        thickness = 0.5.dp
    )
}
