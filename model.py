# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras.layers as tfl

def identity_block(X, f, filters):
  F1, F2, F3 = filters
  X_shortcut = X
  # First inner layer
  X = tfl.Conv2D(filters=F1, kernel_size=1, strides=(1,1), padding="valid")(X)
  X = tfl.BatchNormalization(axis=-1)(X, training=True)
  X = tfl.ReLU()(X)
  # Second inner layer
  X = tfl.Conv2D(filters=F2, kernel_size=f, strides=(1,1), padding="same")(X)
  X = tfl.BatchNormalization(axis=-1)(X, training=True)
  X = tfl.ReLU()(X)
  # Third inner layer
  X = tfl.Conv2D(filters=F3, kernel_size=1, strides=(1,1), padding="valid")(X)
  X = tfl.BatchNormalization(axis=-1)(X, training=True)
  # Identity
  X = tfl.Add()([X, X_shortcut])
  X = tfl.ReLU()(X)
  return X

def convolution_block(X, f, filters, s):
  F1, F2, F3 = filters
  X_shortcut = X
  # First inner layer
  X = tfl.Conv2D(filters=F1, kernel_size=1, strides=(s,s), padding="valid")(X)
  X = tfl.BatchNormalization(axis=-1)(X, training=True)
  X = tfl.ReLU()(X)
  # Second inner layer
  X = tfl.Conv2D(filters=F2, kernel_size=f, strides=(1,1), padding="same")(X)
  X = tfl.BatchNormalization(axis=-1)(X, training=True)
  X = tfl.ReLU()(X)
  # Third inner layer
  X = tfl.Conv2D(filters=F3, kernel_size=1, strides=(1,1), padding="valid")(X)
  X = tfl.BatchNormalization(axis=-1)(X, training=True)
  # Convolution
  X_shortcut = tfl.Conv2D(filters=F3, kernel_size=1, strides=(s,s), padding="valid")(X_shortcut)
  X_shortcut = tfl.BatchNormalization(axis=-1)(X_shortcut, training=True)
  # Identity
  X = tfl.Add()([X, X_shortcut])
  X = tfl.ReLU()(X)
  return X

def resnet(input_shape=(64,64,3), classes=6):
  # Input layer
  X_input = tfl.Input(input_shape)
  # Regular ConvNet
  X = tfl.ZeroPadding2D(padding=(3,3))(X_input)
  X = tfl.Conv2D(filters=64, kernel_size=7, strides=(2,2), padding="valid")(X)
  X = tfl.BatchNormalization(axis=-1)(X, training=True)
  X = tfl.ReLU()(X)
  X = tfl.MaxPooling2D(pool_size=(3,3), strides=(2,2))(X)
  # First block
  X = convolution_block(X, 3, [64, 64, 256], 1)
  X = identity_block(X, 3, [64, 64, 256])
  X = identity_block(X, 3, [64, 64, 256])
  # Second block
  X = convolution_block(X, 3, [128, 128, 512], 2)
  X = identity_block(X, 3, [128, 128, 512])
  X = identity_block(X, 3, [128, 128, 512])
  X = identity_block(X, 3, [128, 128, 512])
  # Third block
  X = convolution_block(X, 3, [256, 256, 1024], 2)
  X = identity_block(X, 3, [256, 256, 1024])
  X = identity_block(X, 3, [256, 256, 1024])
  X = identity_block(X, 3, [256, 256, 1024])
  X = identity_block(X, 3, [256, 256, 1024])
  # Fourth block
  X = convolution_block(X, 3, [512, 512, 2048], 2)
  X = identity_block(X, 3, [512, 512, 2048])
  X = identity_block(X, 3, [512, 512, 2048])
  # Top layers
  X = tfl.AveragePooling2D(pool_size=(2,2))(X)
  X = tfl.Flatten()(X)
  X = tfl.Dense(classes, activation="softmax")(X)
  model = tf.keras.models.Model(inputs=X_input, outputs=X)
  return model