import SwiftUI
import AVFoundation

struct ContentView: View {
    @State private var isSwitchOn: Bool = false
    @State private var isFrontCamera: Bool = false

    var body: some View {
        VStack {
            HStack {
                Button(action: {
                    exit(0)
                }) {
                    Text("Exit")
                }
                Spacer()
                Toggle(isOn: $isSwitchOn) {
                    
                }
            }.padding()
            CameraView(isSwitchOn: $isSwitchOn)
        }
        .navigationBarTitleDisplayMode(.inline)
    }
}

