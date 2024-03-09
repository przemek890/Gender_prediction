import cv2
import numpy as np
import torch
from Src.Gender_net import Net
""""""""""""""""""""""""

import torch.nn as nn
""""""""""""""""""""""""

class Net(nn.Module):
    def __init__(self,dropout_prob=0.5):
        super(Net, self).__init__()

        kernel_s = 3

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=kernel_s, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(25*25*32, 256)

        self.fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = x.reshape(-1, 25*25*32)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.video_capture = cv2.VideoCapture(0)

        self.gender_model = Net()

        checkpoint = torch.load("Models/gender_model.pth")
        self.gender_model.load_state_dict(checkpoint['weights'])
        self.gender_model.eval()


    def detect_faces(self):
        while True:
            _, frame = self.video_capture.read()
            faces = self.face_cascade.detectMultiScale(frame , scaleFactor=1.1, minNeighbors=5, minSize=(50,50))

            for (x, y, w, h) in faces:
                if w > 0 and h > 0:
                    face_tensor = []
                    face_img = frame[y:y + h, x:x + w]
                    face_img = cv2.resize(face_img, (52, 52))
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face_tensor.append(np.asarray(face_img))

                    face_tensor = torch.tensor(np.array(face_tensor) / 255.0, dtype=torch.float32).reshape(-1, 3, 52,52)

                    with torch.no_grad():
                        gender_prediction = self.gender_model(face_tensor)
                        print(gender_prediction)



                    gender_label = "Female" if gender_prediction.item() > 0.5 else "Male"
                    print(gender_label)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    gender_label_pos = (x, y - 10)
                    cv2.putText(frame, gender_label, gender_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                gender_label_pos = (x, y - 10)
                cv2.putText(frame, gender_label, gender_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()


