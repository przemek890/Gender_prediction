import SwiftUI
import UIKit
import AVFoundation
import Vision
import CoreML
// ----------------
class CameraController: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    var captureSession: AVCaptureSession?
    var previewLayer: AVCaptureVideoPreviewLayer?
    var faceRectangleLayer: CAShapeLayer?

    override init() {
        super.init()
        captureSession = AVCaptureSession()

        guard let videoCaptureDevice = AVCaptureDevice.default(for: .video) else { return }
        let videoInput: AVCaptureDeviceInput

        do {
            videoInput = try AVCaptureDeviceInput(device: videoCaptureDevice)
        } catch {
            return
        }

        if captureSession!.canAddInput(videoInput) {
            captureSession!.addInput(videoInput)
        } else {
            return
        }

        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "buffer queue"))
        if captureSession!.canAddOutput(videoOutput) {
            captureSession!.addOutput(videoOutput)
        } else {
            return
        }

        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession!)
        captureSession!.startRunning()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let attachments = CMCopyDictionaryOfAttachments(allocator: kCFAllocatorDefault, target: sampleBuffer, attachmentMode: kCMAttachmentMode_ShouldPropagate) as? [CIImageOption: Any] else { return }

        var ciImage = CIImage(cvPixelBuffer: pixelBuffer, options: attachments)

        // Handle device orientation
        let curDevice = UIDevice.current
        if curDevice.orientation == .landscapeRight {
            ciImage = ciImage.oriented(forExifOrientation: 6)
        } else if curDevice.orientation == .landscapeLeft {
            ciImage = ciImage.oriented(forExifOrientation: 8)
        } else if curDevice.orientation == .portraitUpsideDown {
            ciImage = ciImage.oriented(forExifOrientation: 3)
        }

        let faceDetector = CIDetector(ofType: CIDetectorTypeFace, context: nil, options: [CIDetectorAccuracy: CIDetectorAccuracyHigh])
        let faces = faceDetector?.features(in: ciImage)

        DispatchQueue.main.async {
            self.drawFaceBoxes(faces: faces, in: ciImage.extent.size)
        }
    }


    func drawFaceBoxes(faces: [CIFeature]?, in imageSize: CGSize) {
        guard let faces = faces else { return }
        faceRectangleLayer?.removeFromSuperlayer()
        faceRectangleLayer = CAShapeLayer()
        let faceBoxPath = UIBezierPath()

        for face in faces {
            if let face = face as? CIFaceFeature {
                var faceBox = face.bounds
                // Transform the face box to the preview layer's coordinate space
                let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -imageSize.height)
                faceBox = faceBox.applying(transform)
                faceBoxPath.move(to: faceBox.origin)
                faceBoxPath.append(UIBezierPath(rect: faceBox))
            }
        }

        faceRectangleLayer?.path = faceBoxPath.cgPath
        faceRectangleLayer?.strokeColor = UIColor.green.cgColor
        faceRectangleLayer?.lineWidth = 2
        faceRectangleLayer?.fillColor = UIColor.clear.cgColor

        previewLayer?.addSublayer(faceRectangleLayer!)
    }

    func stopRunning() {
        captureSession?.stopRunning()
    }
}

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
        return viewController
    }

    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
        if isSwitchOn && !(context.coordinator.captureSession?.isRunning ?? false) {
            context.coordinator.captureSession?.startRunning()
        } else if !isSwitchOn && (context.coordinator.captureSession?.isRunning ?? false) {
            context.coordinator.captureSession?.stopRunning()
        }
    }
}
