#!/usr/bin/env python3
"""
File: convert_to_tflite.py
Author: Miguel Gonzalez
Description: Convert a trained mini_XCEPTION .keras model to an int8‐quantized
             TFLite model for Coral Edge TPU, using a representative dataset.
"""

import os
import numpy as np
import tensorflow as tf

from utils.datasets import DataManager, split_data 

# --- 1. Configuración de rutas ---
MODEL_KERAS_PATH = 'trained_models/emotion_models/fer2013_mini_XCEPTION.208-0.61.keras'
OUTPUT_TFLITE_PATH = 'mini_XCEPTION_int8.tflite'

# --- 2. Carga del modelo Keras ---
print(f"Cargando modelo Keras desde {MODEL_KERAS_PATH}")
model = tf.keras.models.load_model(MODEL_KERAS_PATH)

# --- 3. Carga de datos representativos (FER2013) ---
print("Cargando imágenes de entrenamiento para calibración cuantizada…")
data_loader = DataManager('fer2013', image_size=(64, 64))
faces, _ = data_loader.get_data()
faces = faces.astype('float32') / 255.0

# Separamos solo entrenamiento (80%) para el generador representativo
train_images, _ = split_data(faces, None, validation_split=0.2)
train_images = train_images[0] if isinstance(train_images, tuple) else train_images

# --- 4. Generador de datos representativos ---
def representative_data_gen():
    # Toma hasta 100 imágenes para calibrar
    for input_value in train_images[:100]:
        # Añade batch dimension: (1,64,64,1)
        yield [np.expand_dims(input_value, axis=0)]

# --- 5. Conversión a TFLite int8 compatible Edge TPU ---
print("Configurando conversor TFLite para cuantización completa int8…")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Solo operaciones enteras
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Definimos tipos de entrada/salida como uint8 (o int8 según tu canal)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

print("Convirtiendo modelo…")
tflite_model = converter.convert()

# --- 6. Guardado del modelo TFLite ---
with open(OUTPUT_TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)
print(f"Modelo TFLite guardado en {OUTPUT_TFLITE_PATH}")