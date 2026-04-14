import Foundation
import shared

enum SharedBallDetectionBridge {
    private static let entrypoint = BallDetectionEntrypoint.shared

    static var homeState: BallDetectionHomeState {
        entrypoint.homeState()
    }

    static var defaultConfig: BallDetectionScreenConfig {
        entrypoint.defaultConfig()
    }

    static var runtimeStrings: BallDetectionRuntimeStrings {
        entrypoint.runtimeStrings()
    }

    static var appDisplayName: String {
        AppEnvironment.shared.displayName()
    }

    static func resolvedModelPathOrName(config: BallDetectionScreenConfig) -> String {
        entrypoint.resolvedModelPathOrName(config: config)
    }
}
