package com.hermitech.hermivision.domain.inference

import android.graphics.PointF
import com.hermitech.hermivision.data.model.BallFrame
import com.hermitech.hermivision.data.model.CourtMatrix
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f

class CourtProjector {

    /**
     * Project a single ball position from image space to court space.
     *
     * Port of:
     *   ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
     *   ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
     *   → (ball_point[0, 0, 0], ball_point[0, 0, 1])
     *
     * @param ballX  Ball x-coordinate in image space
     * @param ballY  Ball y-coordinate in image space
     * @param homographyInv  Inverse homography matrix (9 floats, row-major 3×3)
     * @return PointF in court reference space, or null if transform fails
     */
    fun projectToCourtCoords(ballX: Float, ballY: Float, homographyInv: FloatArray): PointF? {
        // Build the 3×3 homography matrix
        val hMat = Mat(3, 3, CvType.CV_64F)
        for (r in 0 until 3) {
            for (c in 0 until 3) {
                hMat.put(r, c, homographyInv[r * 3 + c].toDouble())
            }
        }

        // Create source point as MatOfPoint2f (1 point)
        val srcPoint = MatOfPoint2f(
            org.opencv.core.Point(ballX.toDouble(), ballY.toDouble())
        )

        // Transform
        val dstPoint = MatOfPoint2f()
        Core.perspectiveTransform(srcPoint, dstPoint, hMat)

        // Extract result
        val result = if (dstPoint.rows() > 0) {
            val pts = dstPoint.toArray()
            PointF(pts[0].x.toFloat(), pts[0].y.toFloat())
        } else {
            null
        }

        // Release OpenCV Mats
        hMat.release()
        srcPoint.release()
        dstPoint.release()

        return result
    }

    /**
     * Project all bounce positions onto the court reference.
     *
     * Port of main.py loop:
     *   if i in bounces and inv_mat is not None:
     *       ball_point = ball_track[i]
     *       ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
     *
     * @param ballFrames      Full ball trajectory from TrackNet
     * @param courtMatrices   Homography inverse matrices from CourtDetector
     * @param bounceFrameIds  Set of frame IDs where bounces occurred
     * @return List of (frameId, courtPoint) for each bounce that could be projected
     */
    fun projectBounces(ballFrames: List<BallFrame>, courtMatrices: List<CourtMatrix>, bounceFrameIds: Set<Int>): List<Pair<Int, PointF>> {
        val results = mutableListOf<Pair<Int, PointF>>()
        // Build lookup maps for O(1) access
        val ballByFrame = ballFrames.associateBy { it.frameId }
        val courtByFrame = courtMatrices.associateBy { it.frameId }

        for (frameId in bounceFrameIds) {
            val ball = ballByFrame[frameId] ?: continue
            if (ball.x == null || ball.y == null) continue

            val court = courtByFrame[frameId] ?: continue
            val hInv = court.homographyInv ?: continue

            val courtPoint = projectToCourtCoords(ball.x, ball.y, hInv)
            if (courtPoint != null) {
                results.add(Pair(frameId, courtPoint))
            }
        }

        return results.sortedBy { it.first }
    }

    /**
     * Project all ball positions (not just bounces) for trajectory visualization.
     *
     * Useful for drawing the full ball path on the 2D court minimap.
     *
     * @param ballFrames      Full ball trajectory
     * @param courtMatrices   Homography inverse matrices
     * @return List of (frameId, courtPoint) for each frame with valid ball + homography
     */
    fun projectTrajectory(ballFrames: List<BallFrame>, courtMatrices: List<CourtMatrix>): List<Pair<Int, PointF>> {
        val results = mutableListOf<Pair<Int, PointF>>()

        val courtByFrame = courtMatrices.associateBy { it.frameId }

        for (ball in ballFrames) {
            if (ball.x == null || ball.y == null) continue

            val court = courtByFrame[ball.frameId] ?: continue
            val hInv = court.homographyInv ?: continue

            val courtPoint = projectToCourtCoords(ball.x, ball.y, hInv)
            if (courtPoint != null) {
                results.add(Pair(ball.frameId, courtPoint))
            }
        }

        return results
    }
}