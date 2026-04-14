package com.hermitech.hermivision.shared.domain.usecase

import com.hermitech.hermivision.shared.domain.model.BallFrame
import com.hermitech.hermivision.shared.domain.model.YoloDetection
import com.hermitech.hermivision.shared.domain.model.YoloLetterbox

class ParseYoloBallFrameUseCase(
    private val confThreshold: Float = 0.25f,
    private val iouThreshold: Float = 0.45f,
    private val numAnchors: Int = 8400,
    private val maxNmsCandidates: Int = 100,
) {
    operator fun invoke(
        frameId: Int,
        output: Array<FloatArray>,
        letterbox: YoloLetterbox,
    ): BallFrame {
        val candidates = ArrayList<YoloDetection>(maxNmsCandidates)

        for (i in 0 until minOf(numAnchors, output[0].size)) {
            val score = output[4][i]
            if (score > confThreshold) {
                candidates.add(
                    YoloDetection(
                        cx = output[0][i],
                        cy = output[1][i],
                        w = output[2][i],
                        h = output[3][i],
                        score = score,
                    ),
                )
            }
        }

        if (candidates.isEmpty()) {
            return BallFrame(frameId, isVisible = false, x = null, y = null)
        }

        candidates.sortByDescending { it.score }
        val kept = nonMaximumSuppression(candidates)
        val best = kept.firstOrNull() ?: return BallFrame(frameId, isVisible = false, x = null, y = null)

        val rawX = best.cx - letterbox.padX
        val rawY = best.cy - letterbox.padY
        val x = rawX / letterbox.scale
        val y = rawY / letterbox.scale

        return BallFrame(frameId, isVisible = true, x = x, y = y)
    }

    private fun nonMaximumSuppression(candidates: List<YoloDetection>): List<YoloDetection> {
        val kept = ArrayList<YoloDetection>()
        val suppressed = BooleanArray(candidates.size)

        for (i in candidates.indices) {
            if (suppressed[i]) continue

            val anchor = candidates[i]
            kept.add(anchor)

            for (j in i + 1 until candidates.size) {
                if (suppressed[j]) continue
                if (computeIoU(anchor, candidates[j]) > iouThreshold) {
                    suppressed[j] = true
                }
            }
        }

        return kept
    }

    private fun computeIoU(a: YoloDetection, b: YoloDetection): Float {
        val ax1 = a.cx - a.w / 2f
        val ay1 = a.cy - a.h / 2f
        val ax2 = a.cx + a.w / 2f
        val ay2 = a.cy + a.h / 2f

        val bx1 = b.cx - b.w / 2f
        val by1 = b.cy - b.h / 2f
        val bx2 = b.cx + b.w / 2f
        val by2 = b.cy + b.h / 2f

        val ix1 = maxOf(ax1, bx1)
        val iy1 = maxOf(ay1, by1)
        val ix2 = minOf(ax2, bx2)
        val iy2 = minOf(ay2, by2)

        val interW = maxOf(0f, ix2 - ix1)
        val interH = maxOf(0f, iy2 - iy1)
        val interArea = interW * interH

        val aArea = a.w * a.h
        val bArea = b.w * b.h
        val unionArea = aArea + bArea - interArea

        return if (unionArea > 0f) interArea / unionArea else 0f
    }
}
