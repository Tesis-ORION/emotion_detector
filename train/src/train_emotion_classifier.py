"""
 File: train_emotion_classifier.py (modificado para compatibilidad TF)
 Author: Miguel Gonzalez
 Description: Train emotion classification model con guardado .keras y gráficas de métricas y ROC
"""

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.cnn import mini_XCEPTION
from utils.datasets import DataManager, split_data
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# parameters
batch_size = 32
num_epochs = 10000
input_shape = (64, 64, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50
base_path = '../trained_models/emotion_models/'

# Funciones de graficado
def plot_training_history(history):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training Accuracy and Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def plot_roc_curves(model, val_faces, val_emotions):
    y_pred = model.predict(val_faces)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(val_emotions[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves for Each Emotion')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

# Generador de datos
data_generator = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    horizontal_flip=True
)

# Construcción del modelo
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

datasets = ['fer2013']
for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    # Callbacks
    log_file_path = f"{base_path}{dataset_name}_emotion_training.log"
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1)
    trained_models_path = f"{base_path}{dataset_name}_mini_XCEPTION"
    # Guardar en .keras
    model_names = f"{trained_models_path}.{{epoch:02d}}-{{val_accuracy:.2f}}.keras"
    model_checkpoint = ModelCheckpoint(
        filepath=model_names,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
    )
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # Carga de datos
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    faces, emotions = data_loader.get_data()
    faces = faces.astype('float32') / 255.0

    # División train/validation
    train_data, val_data = split_data(faces, emotions, validation_split)
    train_faces, train_emotions = train_data
    val_faces, val_emotions = val_data

    # Entrenamiento
    history = model.fit(
        data_generator.flow(train_faces, train_emotions, batch_size),
        steps_per_epoch=len(train_faces) // batch_size,
        epochs=num_epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=(val_faces, val_emotions)
    )

    # Gráficas finales
    plot_training_history(history)
    plot_roc_curves(model, val_faces, val_emotions)
