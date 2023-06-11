import cv2
import numpy as np
import tensorflow as tf

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.video_capture = cv2.VideoCapture(0)
        self.gender_model = tf.keras.models.load_model('Models/gender_model.h5')
        self.age_model = tf.keras.models.load_model('Models/age_model.h5')

    def detect_faces(self):
        while True:
            ret, frame = self.video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (128, 128))
                face_img = np.expand_dims(face_img, axis=0)
                face_img = np.expand_dims(face_img, axis=-1)
                face_img = face_img / 255.0
                gender_prediction = self.gender_model.predict(face_img)
                gender_label = "Female" if gender_prediction[0] < 0.5 else "Male"  # Swap the gender labels
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                age_prediction = self.age_model.predict(face_img)
                age_label = f"Age: {int(age_prediction[0][0])} years"

                text_size, _ = cv2.getTextSize(age_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                age_label_pos = (x + w - text_size[0], y + h + 30)
                cv2.putText(frame, age_label, age_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 128, 128), 2)

                gender_label_pos = (x, y - 10)
                cv2.putText(frame, gender_label, gender_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 128, 128), 2)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()






