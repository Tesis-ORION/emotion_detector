#!/usr/bin/env python3

import tensorflow as tf

# Verificar si TensorFlow detecta la GPU
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))
