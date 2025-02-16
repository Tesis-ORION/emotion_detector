#!/ur/bin/env python3

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Verificar si TensorFlow detecta la GPU
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))

# Limitar el uso de memoria de la GPU (opcional)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Seleccionar una GPU específica (opcional)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ----------------------------
# Paso 1: Cargar y preprocesar el dataset
# ----------------------------

# Rutas al dataset FER-2013
train_data_dir = 'dataset/train'  # Cambia esto a la ruta de tu dataset de entrenamiento
test_data_dir = 'dataset/test'    # Cambia esto a la ruta de tu dataset de prueba

# Parámetros
img_width, img_height = 48, 48  # Tamaño de las imágenes en FER-2013
batch_size = 64
num_classes = 7  # 7 emociones

# Data augmentation y preprocesamiento
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalizar los valores de píxeles
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Generadores de datos
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',  # FER-2013 es en escala de grises
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical')

# ----------------------------
# Paso 2: Construir el modelo CNN
# ----------------------------

# Crear el modelo
model = Sequential()

# Capa 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Capa 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Capa 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aplanar y conectar a capas densas
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Regularización
model.add(Dense(num_classes, activation='softmax'))  # Salida para 7 emociones

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Resumen del modelo
model.summary()

# ----------------------------
# Paso 3: Entrenar el modelo
# ----------------------------

# Número de épocas
epochs = 50

# Entrenar el modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    epochs=epochs)

# ----------------------------
# Paso 4: Guardar el modelo entrenado
# ----------------------------

model.save('emotion_recognition_model.keras')

# ----------------------------
# Paso 5: Visualizar resultados del entrenamiento
# ----------------------------

# Graficar la precisión y la pérdida
plt.figure(figsize=(12, 4))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión en entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión en validación')
plt.title('Precisión del modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida en entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida en validación')
plt.title('Pérdida del modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.show()
