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
    var sequenceHandler = VNSequenceRequestHandler()

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let attachments = CMCopyDictionaryOfAttachments(allocator: kCFAllocatorDefault, target: sampleBuffer, attachmentMode: kCMAttachmentMode_ShouldPropagate) as? [CIImageOption: Any] else { return }

        var ciImage = CIImage(cvPixelBuffer: pixelBuffer, options: attachments)

        switch UIDevice.current.orientation {
        case .landscapeLeft:
            ciImage = ciImage.oriented(.down)
        case .landscapeRight:
            ciImage = ciImage.oriented(.up)
        case .portraitUpsideDown:
            ciImage = ciImage.oriented(.left)
        default:
            ciImage = ciImage.oriented(.right)
        }

        let faceDetector = CIDetector(ofType: CIDetectorTypeFace, context: nil, options: [CIDetectorAccuracy: CIDetectorAccuracyHigh])
        let faces = faceDetector?.features(in: ciImage)

        DispatchQueue.main.async {
            self.drawFaceBoxes(faces: faces)
        }
    }


    func drawFaceBoxes(faces: [CIFeature]?) {
        guard let faces = faces else { return }
        faceRectangleLayer?.removeFromSuperlayer()
        faceRectangleLayer = CAShapeLayer()
        let faceBoxPath = UIBezierPath()

        for face in faces {
            if let face = face as? CIFaceFeature {
                var faceBox = face.bounds
                let xOffset: CGFloat = 150
                let yOffset: CGFloat = -450
                faceBox.origin.x -= xOffset
                faceBox.origin.y = previewLayer!.frame.height - faceBox.origin.y - faceBox.height - yOffset
                faceBoxPath.move(to: faceBox.origin)
                faceBoxPath.append(UIBezierPath(rect: faceBox))
            }
        }

        faceRectangleLayer?.path = faceBoxPath.cgPath
        faceRectangleLayer?.strokeColor = UIColor.green.cgColor
        faceRectangleLayer?.lineWidth = 5
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
