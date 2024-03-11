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
            var faceBounds = face.bounds
            let hairHeight = faceBounds.height * 0.3
            let hairWidth = faceBounds.width * 0.2
            faceBounds.origin.y -= hairHeight * 0.5
            faceBounds.origin.x -= hairWidth * 0.5
            faceBounds.size.height += hairHeight
            faceBounds.size.width += hairWidth

            let angleInDegrees = 15.0
            let angleInRadians = CGFloat(angleInDegrees * Double.pi / 180.0)

            let ciImageBounds = ciImage.extent
            let midX = ciImageBounds.midX
            let midY = ciImageBounds.midY

            let transform = CGAffineTransform(translationX: midX, y: midY)
                .rotated(by: angleInRadians)
                .translatedBy(x: -midX, y: -midY)

            let rotatedImage = ciImage.transformed(by: transform)

            var faceImage = rotatedImage.cropped(to: faceBounds)


            let context = CIContext(options: nil)
            guard let cgImage = context.createCGImage(faceImage, from: faceImage.extent) else {
                print("Błąd: nie można utworzyć CGImage")
                return
            }
            
            let uiImage = UIImage(cgImage: cgImage)
            
            let size = CGSize(width: 52, height: 52)
            UIGraphicsBeginImageContextWithOptions(size, false, uiImage.scale)
            
            uiImage.draw(in: CGRect(origin: CGPoint.zero, size: size))

            let scaledImage2 = UIGraphicsGetImageFromCurrentImageContext()
            UIGraphicsEndImageContext()

            let ciImage = CIImage(image: scaledImage2!)


            let ciContext = CIContext()
            var pixelBuffer: CVPixelBuffer?
            if let ciImage = ciImage {
                let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(ciImage.extent.width), Int(ciImage.extent.height), kCVPixelFormatType_32BGRA, nil, &pixelBuffer)
                guard status == kCVReturnSuccess else {
                    return
                }

                ciContext.render(ciImage, to: pixelBuffer!)
            } else {
                print("Error: ciImage is nil")
            }

            
            guard let pixelBuffer = pixelBuffer else {
                print("Error: pixelBuffer is nil")
                return
            }
            
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
            

                var dim4 = 0
                var dim3 = 0
                var dim2 = 0
                var dim1 = 0
                
                for y in 0..<52 {
                    for x in 0..<52 {
                        let pixelBase = baseAddress?.advanced(by:  y * bytesPerRow + x * 4)
                        
                        let pix1 = pixelBase?.advanced(by: 2).load(as: UInt8.self)
                        let pix2 = pixelBase?.advanced(by: 1).load(as: UInt8.self)
                        let pix3 = pixelBase?.advanced(by: 0).load(as: UInt8.self)
                        let _  = pixelBase?.advanced(by: 3).load(as: UInt8.self)
                        
                        
                        let pix1Value = Float32(pix1!) / 255.0
                        let pix2Value = Float32(pix2!) / 255.0
                        let pix3Value = Float32(pix3!) / 255.0
 
                        pixelMultiArray[[dim1, dim2, dim3, dim4] as [NSNumber]] = pix1Value as NSNumber
                        dim4 += 1
                        if(dim4 == 52) {dim4 = 0; dim3 += 1}
                        if(dim3 == 52) {dim3 = 0; dim2 += 1}
                        
                        pixelMultiArray[[dim1, dim2, dim3, dim4] as [NSNumber]] = pix2Value as NSNumber
                        dim4 += 1
                        if(dim4 == 52) {dim4 = 0; dim3 += 1}
                        if(dim3 == 52) {dim3 = 0; dim2 += 1}
                        
                        pixelMultiArray[[dim1, dim2, dim3, dim4] as [NSNumber]] = pix3Value as NSNumber
                        dim4 += 1
                        if(dim4 == 52) {dim4 = 0; dim3 += 1}
                        if(dim3 == 52) {dim3 = 0; dim2 += 1}
                    }
                }

                
                CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
                
                let provider = try MLDictionaryFeatureProvider(dictionary: ["x_1": pixelMultiArray])
                
                let prediction = try model?.prediction(from: provider)
                
                let predictionValue = prediction?.featureValue(for: "var_40")?.multiArrayValue?[0]
//                print("Prediction: \(String(describing: predictionValue))")
                
                DispatchQueue.main.async {
                    self.drawFaceBoxes(faces: faces, prediction: predictionValue)
                }
            } catch {
                print("Error making prediction: \(error)")
            }
            
        }
    }
    func drawFaceBoxes(faces: [CIFeature]?, prediction: NSNumber?) {
        guard let faces = faces, let face = faces.first as? CIFaceFeature else { return }
        faceRectangleLayer?.removeFromSuperlayer()
        faceRectangleLayer = CAShapeLayer()
        let faceBoxPath = UIBezierPath()

        var faceBox = face.bounds
        let xOffset: CGFloat = 150
        let yOffset: CGFloat = -450
        faceBox.origin.x -= xOffset
        faceBox.origin.y = previewLayer!.frame.height - faceBox.origin.y - faceBox.height - yOffset
        faceBoxPath.move(to: faceBox.origin)
        faceBoxPath.append(UIBezierPath(rect: faceBox))
        
        let textLayer = CATextLayer()
        textLayer.fontSize = 18
        textLayer.foregroundColor = UIColor.red.cgColor
        let gender = prediction?.floatValue ?? 0 >= 0.5 ? "Female" : "Male"
        let probability = String(format: "%.2f", abs((prediction?.floatValue ?? 0) * 100))
        textLayer.string = "\(gender) (\(probability)%)"
        textLayer.frame = CGRect(x: faceBox.origin.x, y: faceBox.origin.y - 50, width: 200, height: 20)
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
