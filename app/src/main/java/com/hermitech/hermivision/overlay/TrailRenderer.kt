package com.hermitech.hermivision.overlay

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import com.hermitech.hermivision.domain.NormalizedPoint

object TrailRenderer {
    fun DrawScope.drawTrail(trail: List<NormalizedPoint>, color: Color) {
        if (trail.size < 2) {
            return
        }

        val path = Path().apply {
            val first = trail.first()
            moveTo(first.x * size.width, first.y * size.height)
            for (point in trail.drop(1)) {
                lineTo(point.x * size.width, point.y * size.height)
            }
        }

        drawPath(
            path = path,
            color = color.copy(alpha = 0.72f),
            style = Stroke(width = size.minDimension * 0.008f)
        )

        trail.forEachIndexed { index, point ->
            val alpha = (index + 1) / trail.size.toFloat()
            drawCircle(
                color = color.copy(alpha = alpha * 0.8f),
                radius = size.minDimension * 0.008f,
                center = Offset(point.x * size.width, point.y * size.height)
            )
        }
    }
}
