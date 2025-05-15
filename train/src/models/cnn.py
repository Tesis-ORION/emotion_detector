# src/models/cnn.py

from tensorflow.keras.layers import (
    Activation, Conv2D, Dropout, AveragePooling2D, BatchNormalization,
    GlobalAveragePooling2D, Flatten, Input, MaxPooling2D, SeparableConv2D
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2


def simple_CNN(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(16, (7, 7), padding='same',
                     name='image_array', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (7, 7), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Conv2D(32, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(num_classes, (3, 3), padding='same'))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax', name='predictions'))
    return model


def simpler_CNN(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(16, (5, 5), padding='same',
                     name='image_array', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (5, 5), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Conv2D(32, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Conv2D(64, (1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Conv2D(256, (1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(num_classes, (3, 3), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Activation('softmax', name='predictions'))
    return model


def tiny_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    reg = l2(l2_regularization)
    img_input = Input(input_shape)

    # Bloque base
    x = Conv2D(5, (3, 3), kernel_regularizer=reg, use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(5, (3, 3), kernel_regularizer=reg, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Módulo 1
    residual = Conv2D(8, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(8, (3, 3), padding='same',
                        depthwise_regularizer=reg, pointwise_regularizer=reg,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(8, (3, 3), padding='same',
                        depthwise_regularizer=reg, pointwise_regularizer=reg,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Resto módulos similares...
    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Activation('softmax', name='predictions')(x)
    return Model(img_input, outputs)


def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    reg = l2(l2_regularization)
    img_input = Input(input_shape)

    # Bloque base
    x = Conv2D(8, (3, 3), kernel_regularizer=reg, use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), kernel_regularizer=reg, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Módulo 1
    residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        depthwise_regularizer=reg, pointwise_regularizer=reg,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        depthwise_regularizer=reg, pointwise_regularizer=reg,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Módulo 2
    residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        depthwise_regularizer=reg, pointwise_regularizer=reg,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        depthwise_regularizer=reg, pointwise_regularizer=reg,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Módulo 3
    residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        depthwise_regularizer=reg, pointwise_regularizer=reg,\
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        depthwise_regularizer=reg, pointwise_regularizer=reg,\
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Módulo 4
    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        depthwise_regularizer=reg, pointwise_regularizer=reg,\
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        depthwise_regularizer=reg, pointwise_regularizer=reg,\
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Capa final y salida
    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Activation('softmax', name='predictions')(x)
    return Model(img_input, outputs)


def big_XCEPTION(input_shape, num_classes):
    img_input = Input(input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False)(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Activation('softmax', name='predictions')(x)
    return Model(img_input, outputs)


if __name__ == "__main__":
    input_shape = (64, 64, 1)
    num_classes = 7
    model = mini_XCEPTION(input_shape, num_classes)
    model.summary()
