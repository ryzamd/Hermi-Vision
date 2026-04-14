import UIKit

enum BallDetectionScreenFactory {
    static func makeViewController() -> UIViewController {
        NativeBallDetectionViewController(
            config: SharedBallDetectionBridge.defaultConfig,
            runtimeStrings: SharedBallDetectionBridge.runtimeStrings
        )
    }
}
