package com.hermitech.hermivision.inference

class NullBallDetector : BallDetector {
    override fun detect(frameData: FrameData): RawBallCandidate? = null
}
