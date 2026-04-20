import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Model load karo
model = load_model("emotion_model.h5")

# Emotion labels
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# Haar cascade
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_classifier.empty():
    print("ERROR: Haar cascade load nahi hua!")
    exit()

# MacBook camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera nahi khula!")
    exit()

print("Camera chalu ho gaya!")
print("Q dabao band karne ke liye")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame nahi aa raha!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))

        prediction = model.predict(roi, verbose=0)
        label = emotion_labels[prediction.argmax()]

        # Green box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Sirf emotion naam - % nahi
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # MacBook Camera text
    cv2.putText(
        frame,
        "MacBook Camera",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Band ho gaya!")