package com.hermitech.hermivision.domain.inference

/**
 * Static reference data for a standard tennis court.
 *
 * Ported 1:1 from TennisCourtDetector/court_reference.py.
 *
 * All coordinates are in a 2D canvas whose total size is
 * 1665 × 3506 px  (court_width + 2*border_lr  ×  court_height + 2*border_tb).
 */
object CourtReference {
    // Court dimensions (pixels on reference canvas)
    const val COURT_WIDTH = 1117
    const val COURT_HEIGHT = 2408
    const val BORDER_TB = 549        // top & bottom border
    const val BORDER_LR = 274        // left & right  border
    const val TOTAL_WIDTH  = COURT_WIDTH  + BORDER_LR * 2   // 1665
    const val TOTAL_HEIGHT = COURT_HEIGHT + BORDER_TB * 2    // 3506

    // Named line endpoints  (x, y)
    val baselineTop       = arrayOf(floatArrayOf(286f, 561f),  floatArrayOf(1379f, 561f))
    val baselineBottom    = arrayOf(floatArrayOf(286f, 2935f), floatArrayOf(1379f, 2935f))
    val net               = arrayOf(floatArrayOf(286f, 1748f), floatArrayOf(1379f, 1748f))
    val leftCourtLine     = arrayOf(floatArrayOf(286f, 561f),  floatArrayOf(286f, 2935f))
    val rightCourtLine    = arrayOf(floatArrayOf(1379f, 561f), floatArrayOf(1379f, 2935f))
    val leftInnerLine     = arrayOf(floatArrayOf(423f, 561f),  floatArrayOf(423f, 2935f))
    val rightInnerLine    = arrayOf(floatArrayOf(1242f, 561f), floatArrayOf(1242f, 2935f))
    val middleLine        = arrayOf(floatArrayOf(832f, 1110f), floatArrayOf(832f, 2386f))
    val topInnerLine      = arrayOf(floatArrayOf(423f, 1110f), floatArrayOf(1242f, 1110f))
    val bottomInnerLine   = arrayOf(floatArrayOf(423f, 2386f), floatArrayOf(1242f, 2386f))

    /**
     * 14 reference keypoints in the same order the model predicts:
     *   0,1  = baseline_top
     *   2,3  = baseline_bottom
     *   4,5  = left_inner_line
     *   6,7  = right_inner_line
     *   8,9  = top_inner_line
     *  10,11 = bottom_inner_line
     *  12,13 = middle_line
     */
    val keyPoints: Array<FloatArray> = arrayOf(
        baselineTop[0],    baselineTop[1],         // 0, 1
        baselineBottom[0], baselineBottom[1],       // 2, 3
        leftInnerLine[0],  leftInnerLine[1],       // 4, 5
        rightInnerLine[0], rightInnerLine[1],       // 6, 7
        topInnerLine[0],   topInnerLine[1],         // 8, 9
        bottomInnerLine[0], bottomInnerLine[1],     // 10, 11
        middleLine[0],     middleLine[1]            // 12, 13
    )

    /**
     * 12 court configurations, each using 4 keypoint indices.
     * Used to find the best homography by trying all configs and
     * picking the one with the smallest mean back-projection error.
     *
     * Map key = config id (1..12), value = 4 reference points (x,y).
     */
    val courtConf: Map<Int, Array<FloatArray>> = mapOf(
        1  to arrayOf(baselineTop[0], baselineTop[1], baselineBottom[0], baselineBottom[1]),
        2  to arrayOf(leftInnerLine[0], rightInnerLine[0], leftInnerLine[1], rightInnerLine[1]),
        3  to arrayOf(leftInnerLine[0], rightCourtLine[0], leftInnerLine[1], rightCourtLine[1]),
        4  to arrayOf(leftCourtLine[0], rightInnerLine[0], leftCourtLine[1], rightInnerLine[1]),
        5  to arrayOf(topInnerLine[0], topInnerLine[1], bottomInnerLine[0], bottomInnerLine[1]),
        6  to arrayOf(topInnerLine[0], topInnerLine[1], leftInnerLine[1], rightInnerLine[1]),
        7  to arrayOf(leftInnerLine[0], rightInnerLine[0], bottomInnerLine[0], bottomInnerLine[1]),
        8  to arrayOf(rightInnerLine[0], rightCourtLine[0], rightInnerLine[1], rightCourtLine[1]),
        9  to arrayOf(leftCourtLine[0], leftInnerLine[0], leftCourtLine[1], leftInnerLine[1]),
        10 to arrayOf(topInnerLine[0], middleLine[0], bottomInnerLine[0], middleLine[1]),
        11 to arrayOf(middleLine[0], topInnerLine[1], middleLine[1], bottomInnerLine[1]),
        12 to arrayOf(bottomInnerLine[0], bottomInnerLine[1], leftInnerLine[1], rightInnerLine[1])
    )

    /**
     * For each config, the indices into [keyPoints] of the 4 points used.
     * Pre-computed from the Python `court_conf_ind` dict.
     */
    val courtConfIndices: Map<Int, IntArray> by lazy {
        val kpList = keyPoints.toList()
        courtConf.mapValues { (_, confPts) ->
            confPts.map { pt -> kpList.indexOfFirst { it.contentEquals(pt) } }.toIntArray()
        }
    }
}