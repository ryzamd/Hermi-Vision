# iOS Host Scaffold

This folder is the first-step iOS host scaffold for the KMP migration.

## What is included

- `project.yml` for generating an Xcode project with XcodeGen
- a small SwiftUI host app
- `BallDetectionScreenFactory` as the native entrypoint that will later mount `YOLOView`
- `Info.plist` with camera usage text already declared for the future realtime detection screen

## Generate the Xcode project

1. Install XcodeGen if it is not available on your machine.
2. From this folder, run `xcodegen generate`.
3. Open the generated `iosApp.xcodeproj`.
4. Build the shared Kotlin framework first with Gradle when wiring the local framework path.

## Next step for Ball Detection v1

- add the exported `best.mlpackage` to the iOS app target bundle
- replace the placeholder `NativeBallDetectionViewController` content with the reused `YOLOView` flow
- keep camera/session teardown owned by that screen controller so repeated open/close cycles do not leak resources
