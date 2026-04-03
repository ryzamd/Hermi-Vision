package com.hermitech.hermivision.ui.results

import android.graphics.PointF
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.hermitech.hermivision.domain.inference.CourtReference

private val CourtGreen = Color(0xFF2E7D32)
private val CourtLinePaint = Color(0xFFFFFFFF)
private val CourtBorder = Color(0xFF1B5E20)
private val BounceColor = Color(0xFFFF6B35)
private val TrajectoryColor = Color(0xFF66BB6A)
private val NetColor = Color(0xFFBDBDBD)

@Composable
fun CourtMatrix2DView(bouncePoints: List<Pair<Int, PointF>>, trajectoryPoints: List<Pair<Int, PointF>>?, modifier: Modifier = Modifier) {
    val courtAspect = CourtReference.TOTAL_WIDTH.toFloat() / CourtReference.TOTAL_HEIGHT.toFloat()

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Court View",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onSurface
        )

        Spacer(modifier = Modifier.height(4.dp))

        LegendRow()

        Spacer(modifier = Modifier.height(8.dp))
        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(courtAspect),
            color = CourtBorder,
            shape = MaterialTheme.shapes.medium,
            tonalElevation = 4.dp
        ) {
            Canvas(modifier = Modifier.fillMaxSize()) {
                val scaleX = size.width / CourtReference.TOTAL_WIDTH.toFloat()
                val scaleY = size.height / CourtReference.TOTAL_HEIGHT.toFloat()

                drawCourtSurface(scaleX, scaleY)
                drawCourtLines(scaleX, scaleY)
                drawNet(scaleX, scaleY)

                if (!trajectoryPoints.isNullOrEmpty()) {
                    drawTrajectory(trajectoryPoints, scaleX, scaleY)
                }
                for ((_, pt) in bouncePoints) {
                    val cx = pt.x * scaleX
                    val cy = pt.y * scaleY
                    drawCircle(
                        color = BounceColor.copy(alpha = 0.3f),
                        radius = 12f,
                        center = Offset(cx, cy)
                    )
                    drawCircle(
                        color = BounceColor,
                        radius = 6f,
                        center = Offset(cx, cy)
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            text = "${bouncePoints.size} bounce${if (bouncePoints.size != 1) "s" else ""} detected",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
        )
    }
}

@Composable
private fun LegendRow() {
    Row(
        horizontalArrangement = Arrangement.spacedBy(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        LegendItem(color = BounceColor, label = "Bounce")
        LegendItem(color = TrajectoryColor, label = "Trajectory")
    }
}

@Composable
private fun LegendItem(color: Color, label: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Canvas(modifier = Modifier.size(10.dp)) {
            drawCircle(color = color, radius = size.minDimension / 2)
        }
        Text(
            text = label,
            fontSize = 11.sp,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
        )
    }
}

private fun DrawScope.drawCourtSurface(scaleX: Float, scaleY: Float) {
    val left = CourtReference.BORDER_LR.toFloat() * scaleX
    val top = CourtReference.BORDER_TB.toFloat() * scaleY
    val right = (CourtReference.BORDER_LR + CourtReference.COURT_WIDTH).toFloat() * scaleX
    val bottom = (CourtReference.BORDER_TB + CourtReference.COURT_HEIGHT).toFloat() * scaleY

    drawRect(
        color = CourtGreen,
        topLeft = Offset(left, top),
        size = androidx.compose.ui.geometry.Size(right - left, bottom - top)
    )
}

private fun DrawScope.drawCourtLines(scaleX: Float, scaleY: Float) {

    fun drawLine(from: FloatArray, to: FloatArray) {
        drawLine(
            color = CourtLinePaint,
            start = Offset(from[0] * scaleX, from[1] * scaleY),
            end = Offset(to[0] * scaleX, to[1] * scaleY),
            strokeWidth = 2f,
            cap = StrokeCap.Round
        )
    }

    // Court boundary lines
    drawLine(CourtReference.baselineTop[0], CourtReference.baselineTop[1])
    drawLine(CourtReference.baselineBottom[0], CourtReference.baselineBottom[1])
    drawLine(CourtReference.leftCourtLine[0], CourtReference.leftCourtLine[1])
    drawLine(CourtReference.rightCourtLine[0], CourtReference.rightCourtLine[1])

    // Inner lines (service boxes)
    drawLine(CourtReference.leftInnerLine[0], CourtReference.leftInnerLine[1])
    drawLine(CourtReference.rightInnerLine[0], CourtReference.rightInnerLine[1])
    drawLine(CourtReference.topInnerLine[0], CourtReference.topInnerLine[1])
    drawLine(CourtReference.bottomInnerLine[0], CourtReference.bottomInnerLine[1])

    // Center service line
    drawLine(CourtReference.middleLine[0], CourtReference.middleLine[1])
}

private fun DrawScope.drawNet(scaleX: Float, scaleY: Float) {
    // Net line — slightly thicker, different color
    drawLine(
        color = NetColor,
        start = Offset(CourtReference.net[0][0] * scaleX, CourtReference.net[0][1] * scaleY),
        end = Offset(CourtReference.net[1][0] * scaleX, CourtReference.net[1][1] * scaleY),
        strokeWidth = 3f,
        cap = StrokeCap.Round
    )
}

private fun DrawScope.drawTrajectory(points: List<Pair<Int, PointF>>, scaleX: Float, scaleY: Float) {
    if (points.size < 2) return

    val path = Path().apply {
        val first = points.first().second
        moveTo(first.x * scaleX, first.y * scaleY)
        for (i in 1 until points.size) {
            val pt = points[i].second
            lineTo(pt.x * scaleX, pt.y * scaleY)
        }
    }

    drawPath(
        path = path,
        color = TrajectoryColor.copy(alpha = 0.5f),
        style = Stroke(width = 1.5f, cap = StrokeCap.Round)
    )
}