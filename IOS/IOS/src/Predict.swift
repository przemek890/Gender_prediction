import UIKit
import AVFoundation
import Vision
import CoreML
//------------
func Predict(faceImage: CIImage , ciImage: CIImage, model: MLModel?) -> NSNumber {

    let context = CIContext(options: nil)
    guard let cgImage = context.createCGImage(faceImage, from: faceImage.extent) else {
        print("Błąd: nie można utworzyć CGImage")
        return 0.0
    }
    
    let uiImage = UIImage(cgImage: cgImage)
    
    let size = CGSize(width: 100, height: 100)
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
            return 0.0
        }

        ciContext.render(ciImage, to: pixelBuffer!)
    } else {
        print("Error: ciImage is nil")
    }

    
    guard let pixelBuffer = pixelBuffer else {
        print("Error: pixelBuffer is nil")
        return 0.0
    }
    
//            let width = CVPixelBufferGetWidth(pixelBuffer)
//            let height = CVPixelBufferGetHeight(pixelBuffer)
//            print("Image dimensions: \(width) x \(height)")
//

    CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
    let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    
    do {
        let pixelMultiArray = try MLMultiArray(shape: [1, 3, 100, 100], dataType: .float32)
    

        var dim4 = 0
        var dim3 = 0
        var dim2 = 0
        var dim1 = 0
        
        for y in 0..<100 {
            for x in 0..<100 {
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
                if(dim4 == 100) {dim4 = 0; dim3 += 1}
                if(dim3 == 100) {dim3 = 0; dim2 += 1}
                
                pixelMultiArray[[dim1, dim2, dim3, dim4] as [NSNumber]] = pix2Value as NSNumber
                dim4 += 1
                if(dim4 == 100) {dim4 = 0; dim3 += 1}
                if(dim3 == 100) {dim3 = 0; dim2 += 1}
                
                pixelMultiArray[[dim1, dim2, dim3, dim4] as [NSNumber]] = pix3Value as NSNumber
                dim4 += 1
                if(dim4 == 100) {dim4 = 0; dim3 += 1}
                if(dim3 == 100) {dim3 = 0; dim2 += 1}
            }
        }

        
        CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        let provider = try MLDictionaryFeatureProvider(dictionary: ["x_1": pixelMultiArray])
        
        let prediction = try model?.prediction(from: provider)
        
        let predictionValue = prediction?.featureValue(for: "var_40")?.multiArrayValue?[0]
//                print("Prediction: \(String(describing: predictionValue))")
    
        return predictionValue ?? 0.0
        
    } catch {
        print("Error making prediction: \(error)")
    }
return 0.0
}
