#!/usr/bin/env python3

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ----------------------------
# Paso 1: Cargar el modelo entrenado
# ----------------------------

# Cargar el modelo guardado
model = load_model('emotion_recognition_model.keras')

# Etiquetas de emociones
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ----------------------------
# Paso 2: Configurar la cámara web
# ----------------------------

# Iniciar la captura de video
cap = cv2.VideoCapture(0)  # 0 para la cámara predeterminada

# Cargar el clasificador de rostros de OpenCV
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ----------------------------
# Paso 3: Detección de emociones en tiempo real
# ----------------------------

while True:
    # Capturar un fotograma de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en el fotograma
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Procesar cada rostro detectado
    for (x, y, w, h) in faces:
        # Recortar la región del rostro
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))  # Redimensionar a 48x48 (tamaño esperado por el modelo)
        face = face.astype('float32') / 255.0  # Normalizar
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)  # Añadir dimensión del batch

        # Predecir la emoción
        predictions = model.predict(face)
        emotion = emotion_labels[np.argmax(predictions)]

        # Dibujar un rectángulo alrededor del rostro y mostrar la emoción
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar el fotograma con las detecciones
    cv2.imshow('Emotion Detection', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
