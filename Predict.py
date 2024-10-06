import cv2
import numpy as np
from tensorflow.keras.models import load_model

def detect_emotion_from_webcam(model_path, class_labels):
    model = load_model(model_path)

    print(f"Class labels: {class_labels}")  # Tambahkan ini untuk melihat label yang digunakan

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48))
            face_normalized = face_resized / 255.0
            face_array = np.expand_dims(face_normalized, axis=[0, -1])

            predictions = model.predict(face_array)
            predicted_class = np.argmax(predictions)

            # Tambahkan print ini untuk debugging
            print(f"Predictions: {predictions}, Predicted class index: {predicted_class}")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, class_labels[predicted_class], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()