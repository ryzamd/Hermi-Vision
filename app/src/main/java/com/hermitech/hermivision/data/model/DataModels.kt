package com.hermitech.hermivision.data.model

data class BallFrame(
    val frameId: Int,
    val isVisible: Boolean,
    val x: Float?,
    val y: Float?
)

data class BounceEvent(
    val frameId: Int,
    val x: Float,
    val y: Float
)

data class CourtMatrix(val frameId: Int, val homographyInv: FloatArray?) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as CourtMatrix

        if (frameId != other.frameId) return false
        if (homographyInv != null) {
            if (other.homographyInv == null) return false
            if (!homographyInv.contentEquals(other.homographyInv)) return false
        } else if (other.homographyInv != null) return false

        return true
    }

    override fun hashCode(): Int {
        var result = frameId
        result = 31 * result + (homographyInv?.contentHashCode() ?: 0)
        return result
    }
}