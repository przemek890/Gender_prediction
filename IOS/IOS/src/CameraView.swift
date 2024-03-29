import SwiftUI
import UIKit
import AVFoundation
// ----------------

struct CameraView: UIViewControllerRepresentable {
    @Binding var isSwitchOn: Bool

    func makeCoordinator() -> Coordinator {
        return Coordinator(self)
    }

    class Coordinator: CameraController {
        var parent: CameraView

        init(_ parent: CameraView) {
            self.parent = parent
            super.init()
        }

    }

    func makeUIViewController(context: Context) -> UIViewController {
        let viewController = UIViewController()
        let previewLayer = AVCaptureVideoPreviewLayer(session: context.coordinator.captureSession!)
        previewLayer.frame = viewController.view.layer.bounds
        previewLayer.videoGravity = .resizeAspectFill
        viewController.view.layer.addSublayer(previewLayer)
        context.coordinator.previewLayer = previewLayer

        if !isSwitchOn {
            context.coordinator.captureSession?.stopRunning()
        }

        return viewController
    }

    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
        DispatchQueue.global(qos: .userInitiated).async {
            if self.isSwitchOn && !(context.coordinator.captureSession?.isRunning ?? false) {
                context.coordinator.captureSession?.startRunning()
            } else if !self.isSwitchOn && (context.coordinator.captureSession?.isRunning ?? false) {
                context.coordinator.captureSession?.stopRunning()
            }
        }
    }
}
