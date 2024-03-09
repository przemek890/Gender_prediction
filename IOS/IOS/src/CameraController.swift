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
            let desiredWidth = 52.0
            let desiredHeight = 52.0
            let scaleX = desiredWidth / faceImage.extent.width
            let scaleY = desiredHeight / faceImage.extent.height
            var scaledImage = faceImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            
            // Przekształć CIImage na CVPixelBuffer
            let ciContext = CIContext()
            var pixelBuffer: CVPixelBuffer?
            let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(scaledImage.extent.width), Int(scaledImage.extent.height), kCVPixelFormatType_32BGRA, nil, &pixelBuffer)
            guard status == kCVReturnSuccess else {
                return
            }
            
            ciContext.render(scaledImage, to: pixelBuffer!)
            
            // Sprawdź, czy pixelBuffer nie jest nil
            guard let pixelBuffer = pixelBuffer else {
                print("Error: pixelBuffer is nil")
                return
            }
            
            // Wydrukuj wymiary obrazu
            let width = CVPixelBufferGetWidth(pixelBuffer)
            let height = CVPixelBufferGetHeight(pixelBuffer)
            print("Image dimensions: \(width) x \(height)")
            
            // Pobierz wartość każdego piksela
            CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
            let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            
            do {
                // Utwórz MLMultiArray z tablicy pikseli
                let pixelMultiArray = try MLMultiArray(shape: [1, 3, 52, 52], dataType: .float32)
                
                for y in 0..<52 {
                    for x in 0..<52 {
                        let pixelBase = baseAddress?.advanced(by: y * bytesPerRow + x * 4)
                        
                        let redPixel = pixelBase?.advanced(by: 1).load(as: UInt8.self)
                        let greenPixel = pixelBase?.advanced(by: 2).load(as: UInt8.self)
                        let bluePixel = pixelBase?.advanced(by: 3).load(as: UInt8.self)
                        
                        let redValue = Float32(redPixel!) / 255.0
                        let greenValue = Float32(greenPixel!) / 255.0
                        let blueValue = Float32(bluePixel!) / 255.0
                        
                        // Umieść wartości pikseli w MLMultiArray
                        pixelMultiArray[[0, 0, y, x] as [NSNumber]] = redValue as NSNumber
                        pixelMultiArray[[0, 1, y, x] as [NSNumber]] = greenValue as NSNumber
                        pixelMultiArray[[0, 2, y, x] as [NSNumber]] = blueValue as NSNumber
                    }
                }
                
                CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
                
                let provider = try MLDictionaryFeatureProvider(dictionary: ["x_1": pixelMultiArray])
                
                let prediction = try model?.prediction(from: provider)
                
                let predictionValue = prediction?.featureValue(for: "var_40")?.multiArrayValue?[0]
                print("Prediction: \(String(describing: predictionValue))")
                
                DispatchQueue.main.async {
                    self.drawFaceBoxes(faces: faces, prediction: predictionValue)
                }
            } catch {
                print("Error making prediction: \(error)")
            }
            
        }
    }

    func drawFaceBoxes(faces: [CIFeature]?, prediction: NSNumber?) {
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
                
                // Dodaj tekst do ramki
                let textLayer = CATextLayer()
                textLayer.fontSize = 18
                textLayer.foregroundColor = UIColor.red.cgColor
                textLayer.string = prediction?.floatValue ?? 0 >= 0.5 ? "Female" : "Male"
                textLayer.frame = CGRect(x: faceBox.origin.x, y: faceBox.origin.y - 50, width: 100, height: 20)
                faceRectangleLayer?.addSublayer(textLayer)
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
