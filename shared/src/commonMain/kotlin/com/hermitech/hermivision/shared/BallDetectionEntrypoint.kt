package com.hermitech.hermivision.shared

data class BallDetectionHomeState(
    val appName: String,
    val screenTitle: String,
    val headline: String,
    val description: String,
    val primaryActionTitle: String,
)

data class BallDetectionScreenConfig(
    val modelName: String = "best",
    val modelPathOverride: String? = null,
)

data class BallDetectionRuntimeStrings(
    val closeButtonTitle: String = "Close",
    val requestingPermissionMessage: String = "Requesting camera permission...",
    val permissionDeniedMessage: String = "Camera permission was denied. Enable it in Settings to use realtime ball detection.",
    val permissionUnavailableMessage: String = "Camera permission is unavailable. Enable camera access in Settings to continue.",
    val permissionUnknownMessage: String = "Camera permission is unavailable due to an unknown system state.",
    val missingModelMessage: String = "Model configuration is missing. Add best.mlpackage to the app bundle or set YOLOModelName / YOLOModelPath.",
    val loadingModelMessage: String = "Loading model...",
    val modelLoadFailurePrefix: String = "Unable to load model.",
)

object BallDetectionEntrypoint {
    fun homeState(): BallDetectionHomeState = BallDetectionHomeState(
        appName = AppEnvironment.displayName(),
        screenTitle = "Home",
        headline = AppEnvironment.displayName(),
        description = "This iOS host now reads its launch copy and detection configuration from the shared Kotlin Multiplatform entrypoint.",
        primaryActionTitle = "Open Ball Detection",
    )

    fun defaultConfig(): BallDetectionScreenConfig = BallDetectionScreenConfig()

    fun runtimeStrings(): BallDetectionRuntimeStrings = BallDetectionRuntimeStrings()

    fun resolvedModelPathOrName(config: BallDetectionScreenConfig = defaultConfig()): String =
        config.modelPathOverride?.takeIf { it.isNotBlank() } ?: config.modelName
}
