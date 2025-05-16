#!/usr/bin/env python3
"""
Real-time emotion recognition on Coral Edge TPU
using a compiled TFLite model and OpenCV Haar cascades.
"""

import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import set_input, input_size
from pycoral.adapters.classify import get_classes  # actual signature usa score_threshold

# 1. Paths
MODEL_PATH = 'mini_XCEPTION_int8_edgetpu.tflite'
HAAR_PATH  = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# 2. Load interpreter
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()
in_width, in_height = input_size(interpreter)

# 3. Emotion labels
EMOTION_LABELS = [
    'Angry', 'Disgust', 'Fear',
    'Happy', 'Sad', 'Surprise', 'Neutral'
]

# 4. Init face detector & camera
face_cascade = cv2.CascadeClassifier(HAAR_PATH)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )

        for (x, y, w, h) in faces:
            # Extract ROI & preprocess
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (in_width, in_height))
            roi = np.expand_dims(roi, axis=2)  # (h, w, 1)
            set_input(interpreter, roi)
            interpreter.invoke()

            # Get top class (use score_threshold instead of threshold)
            classes = get_classes(interpreter, top_k=1, score_threshold=0.0)
            if classes:
                cls   = classes[0]
                label = EMOTION_LABELS[cls.id]
                score = cls.score  # [0.0–1.0]

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label}: {score*100:.1f}%",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

        cv2.imshow('Coral Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
