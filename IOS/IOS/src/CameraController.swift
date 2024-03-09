import UIKit
import AVFoundation
import Vision
import CoreML
//------------
class CameraController: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    var captureSession: AVCaptureSession?
    var previewLayer: AVCaptureVideoPreviewLayer?
    var faceRectangleLayer: CAShapeLayer?
    var model: MLModel?
    var scaledImage: CIImage?

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
        
        // Załaduj model do predykcji płci
        guard let modelURL = Bundle.main.url(forResource: "gender_model", withExtension: "mlmodelc") else {
            fatalError("Failed to find model file.")
        }
        do {
            let mlModel = try MLModel(contentsOf: modelURL)
            model = mlModel
        } catch {
            print("Error loading model: \(error)")
        }
        
        // Przeskaluj obraz do 52x52 pikseli
        if let face = faces?.first as? CIFaceFeature {
            let faceBounds = face.bounds
            let faceImage = ciImage.cropped(to: faceBounds)
            let scaleX = 52 / faceImage.extent.width
            let scaleY = 52 / faceImage.extent.height
            let scaledImage = faceImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            
            // Przekształć obraz na RGB
            let rgbImage = scaledImage.applyingFilter("CIColorControls", parameters: [
                kCIInputSaturationKey: 1,
                kCIInputBrightnessKey: 0,
                kCIInputContrastKey: 1
            ])
            
            // Normalizuj wartości pikseli
            let normalizedImage = rgbImage.applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: 1/255.0, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: 1/255.0, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: 1/255.0, w: 0),
                "inputBiasVector": CIVector(x: 0, y: 0, z: 0, w: 0)
            ])
            
            // Przekształć CIImage na CVPixelBuffer
            let ciContext = CIContext()
            var pixelBuffer: CVPixelBuffer?
            let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(normalizedImage.extent.width), Int(normalizedImage.extent.height), kCVPixelFormatType_32ARGB, nil, &pixelBuffer)
            guard status == kCVReturnSuccess else {
                return
            }
            ciContext.render(normalizedImage, to: pixelBuffer!)
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
