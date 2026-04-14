import AVFoundation
import UIKit
import YOLO
import shared

final class NativeBallDetectionViewController: UIViewController {
    private let closeButton = UIButton(type: .system)
    private let statusLabel = UILabel()
    private let activityIndicator = UIActivityIndicatorView(style: .large)
    private let config: BallDetectionScreenConfig
    private let runtimeStrings: BallDetectionRuntimeStrings
    private var yoloView: YOLOView?

    init(
        config: BallDetectionScreenConfig,
        runtimeStrings: BallDetectionRuntimeStrings
    ) {
        self.config = config
        self.runtimeStrings = runtimeStrings
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        title = SharedBallDetectionBridge.appDisplayName
        edgesForExtendedLayout = [.top, .bottom, .left, .right]
        extendedLayoutIncludesOpaqueBars = true
        configureUI()
        configureCloseButton()
        checkCameraAuthorizationAndStart()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        yoloView?.resume()
    }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        // Stop the capture session as soon as this screen leaves the foreground.
        yoloView?.stop()
    }

    private func configureUI() {
        view.backgroundColor = .black

        activityIndicator.translatesAutoresizingMaskIntoConstraints = false
        activityIndicator.hidesWhenStopped = true

        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        statusLabel.font = .preferredFont(forTextStyle: .body)
        statusLabel.textColor = .white
        statusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.55)
        statusLabel.numberOfLines = 0
        statusLabel.textAlignment = .center
        statusLabel.layer.cornerRadius = 14
        statusLabel.layer.masksToBounds = true

        closeButton.translatesAutoresizingMaskIntoConstraints = false

        view.addSubview(activityIndicator)
        view.addSubview(statusLabel)
        view.addSubview(closeButton)

        NSLayoutConstraint.activate([
            closeButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            closeButton.leadingAnchor.constraint(equalTo: view.layoutMarginsGuide.leadingAnchor),

            activityIndicator.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            activityIndicator.centerYAnchor.constraint(equalTo: view.centerYAnchor),

            statusLabel.leadingAnchor.constraint(equalTo: view.layoutMarginsGuide.leadingAnchor),
            statusLabel.trailingAnchor.constraint(equalTo: view.layoutMarginsGuide.trailingAnchor),
            statusLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -24),
            statusLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
        ])
    }

    private func configureCloseButton() {
        var configuration = UIButton.Configuration.filled()
        configuration.title = runtimeStrings.closeButtonTitle
        configuration.baseForegroundColor = .white
        configuration.baseBackgroundColor = UIColor.black.withAlphaComponent(0.45)
        closeButton.configuration = configuration
        closeButton.addTarget(self, action: #selector(closeTapped), for: .touchUpInside)
    }

    private func checkCameraAuthorizationAndStart() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            mountYOLOView()
        case .notDetermined:
            showStatus(runtimeStrings.requestingPermissionMessage)
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    guard let self else { return }
                    if granted {
                        self.mountYOLOView()
                    } else {
                        self.showStatus(self.runtimeStrings.permissionDeniedMessage)
                    }
                }
            }
        case .denied, .restricted:
            showStatus(runtimeStrings.permissionUnavailableMessage)
        @unknown default:
            showStatus(runtimeStrings.permissionUnknownMessage)
        }
    }

    private func mountYOLOView() {
        if let yoloView {
            activityIndicator.stopAnimating()
            statusLabel.isHidden = true
            yoloView.resume()
            return
        }

        guard let modelPathOrName = resolveModelPathOrName() else {
            showStatus(runtimeStrings.missingModelMessage)
            return
        }

        activityIndicator.startAnimating()
        showStatus(runtimeStrings.loadingModelMessage)

        let yoloView = YOLOView(frame: view.bounds)
        yoloView.translatesAutoresizingMaskIntoConstraints = false
        self.yoloView = yoloView
        view.insertSubview(yoloView, at: 0)

        NSLayoutConstraint.activate([
            yoloView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            yoloView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            yoloView.topAnchor.constraint(equalTo: view.topAnchor),
            yoloView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])

        yoloView.setModel(modelPathOrName: modelPathOrName, task: .detect) { [weak self] result in
            guard let self else { return }
            DispatchQueue.main.async {
                self.activityIndicator.stopAnimating()
                switch result {
                case .success:
                    self.statusLabel.isHidden = true
                case .failure(let error):
                    self.showStatus("\(self.runtimeStrings.modelLoadFailurePrefix)\n\(error.localizedDescription)\nResolved value: \(modelPathOrName)")
                }
            }
        }
    }

    private func resolveModelPathOrName() -> String? {
        let resolved = SharedBallDetectionBridge.resolvedModelPathOrName(config: config)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return resolved.isEmpty ? nil : resolved
    }

    private func showStatus(_ message: String) {
        statusLabel.isHidden = false
        statusLabel.text = "  \(message)  "
    }

    @objc
    private func closeTapped() {
        dismiss(animated: true)
    }
}
