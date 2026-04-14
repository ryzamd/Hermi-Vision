import SwiftUI

struct ContentView: View {
    @State private var isPresentingBallDetection = false
    private let homeState = SharedBallDetectionBridge.homeState

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                Text(homeState.headline)
                    .font(.largeTitle.weight(.bold))

                Text(homeState.description)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)

                Button(homeState.primaryActionTitle) {
                    isPresentingBallDetection = true
                }
                .buttonStyle(.borderedProminent)
            }
            .padding(24)
            .navigationTitle(homeState.screenTitle)
            .fullScreenCover(isPresented: $isPresentingBallDetection) {
                BallDetectionContainerView()
                    .ignoresSafeArea()
                    .background(.black)
            }
        }
    }
}

#Preview {
    ContentView()
}
