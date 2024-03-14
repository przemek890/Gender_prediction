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
        var faces = faceDetector?.features(in: ciImage) as? [CIFaceFeature]

        let straightFaces = faces?.filter { face in
            abs(face.faceAngle) < 10
        }
        
        faces = straightFaces
        
        guard let modelURL = Bundle.main.url(forResource: "gender_model", withExtension: "mlmodelc") else {
            fatalError("Failed to find model file.")
        }
        do {
            let mlModel = try MLModel(contentsOf: modelURL)
            model = mlModel
        } catch {
            print("Error loading model: \(error)")
        }
        
        let predictionValue = Predict(faces:faces, ciImage: ciImage, model: model)
        
        DispatchQueue.main.async {
            self.drawFaceBoxes(faces: faces, prediction: predictionValue)
        }
    
    }
    func drawFaceBoxes(faces: [CIFeature]?, prediction: NSNumber?) {
        guard let faces = faces, let face = faces.first as? CIFaceFeature else { return }
        faceRectangleLayer?.removeFromSuperlayer()
        faceRectangleLayer = CAShapeLayer()
        let faceBoxPath = UIBezierPath()

        var faceBox = face.bounds
        var xOffset: Double
        var yOffset: Double
        if UIDevice.current.userInterfaceIdiom == .pad {
            xOffset = 125
            yOffset = -425
        } else if UIDevice.current.userInterfaceIdiom == .phone {
            xOffset = 0.0 // TODO
            yOffset = 0.0 // TODO
        }
        else {
            xOffset = 0.0
            yOffset = 0.0
        }
        
        faceBox.origin.x -= xOffset
        faceBox.origin.y = previewLayer!.frame.height - faceBox.origin.y - faceBox.height - yOffset
        faceBoxPath.move(to: faceBox.origin)
        faceBoxPath.append(UIBezierPath(rect: faceBox))
        
        let textLayer = CATextLayer()
        textLayer.fontSize = 18
        textLayer.foregroundColor = UIColor.white.cgColor
        let gender = prediction?.floatValue ?? 1.0 >= 0.5 ? "Female" : "Male"
        var probability: String
        if gender == "Male" {
            probability = String(format: "%.2f", abs((1.0 - (prediction?.floatValue ?? 1.0)) * 100))
        }
        else {
            probability = String(format: "%.2f", abs((prediction?.floatValue ?? 0.0) * 100))
        }
        textLayer.string = "\(gender) (\(probability)%)"
        textLayer.frame = CGRect(x: faceBox.origin.x, y: faceBox.origin.y - 30, width: 200, height: 20)

        textLayer.shadowColor = UIColor.black.cgColor
        textLayer.shadowOffset = CGSize(width: 0, height: 0)
        textLayer.shadowOpacity = 1
        textLayer.shadowRadius = 2

        faceRectangleLayer?.addSublayer(textLayer)

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
