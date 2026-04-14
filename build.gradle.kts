import java.io.File

// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.multiplatform) apply false
    alias(libs.plugins.kotlin.compose) apply false
}

val localBuildRoot = File(System.getProperty("java.io.tmpdir"), "hermivision-gradle")

allprojects {
    val projectBuildPath = if (path == ":") {
        "root"
    } else {
        path.removePrefix(":").replace(':', '/')
    }

    // Keep Gradle build output on the internal disk to avoid macOS AppleDouble files on the external drive.
    layout.buildDirectory.set(localBuildRoot.resolve(projectBuildPath))
}
