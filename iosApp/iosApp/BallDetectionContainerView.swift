import SwiftUI

struct BallDetectionContainerView: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> UIViewController {
        BallDetectionHostViewController()
    }

    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
    }
}

final class BallDetectionHostViewController: UIViewController {
    private let contentViewController = BallDetectionScreenFactory.makeViewController()

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        view.insetsLayoutMarginsFromSafeArea = false
        additionalSafeAreaInsets = .zero

        addChild(contentViewController)
        contentViewController.view.translatesAutoresizingMaskIntoConstraints = false
        contentViewController.view.backgroundColor = .black
        view.addSubview(contentViewController.view)

        NSLayoutConstraint.activate([
            contentViewController.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            contentViewController.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            contentViewController.view.topAnchor.constraint(equalTo: view.topAnchor),
            contentViewController.view.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])

        contentViewController.didMove(toParent: self)
    }
}
