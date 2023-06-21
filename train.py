# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.utils import image_dataset_from_directory

from tensorflow.keras.preprocessing import image
import os

import model

# create and compile the model
model = resnet(input_shape=(64,64,3), classes=6)

model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)

# prepare the datasets
seed = 42
batch_size = 32
image_size = (64,64)
train_dataset = image_dataset_from_directory(
    "dataset/seg_train/seg_train",
    shuffle=True,
    batch_size=batch_size,
    image_size=image_size,
    validation_split=0.2,
    subset="training",
    seed=seed
)
val_dataset = image_dataset_from_directory(
    "dataset/seg_train/seg_train",
    shuffle=True,
    batch_size=batch_size,
    image_size=image_size,
    validation_split=0.2,
    subset="validation",
    seed=seed
)
test_dataset = image_dataset_from_directory(
    "dataset/seg_test/seg_test",
    shuffle=True,
    batch_size=batch_size,
    image_size=image_size,
    seed=seed
)

# prefetch the training dataset
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

# one-hot encode the datasets
n_class = 6

def one_hot_in_dataset(images, labels):
  labels = tf.one_hot(labels, n_class)
  return images, labels

def norm_in_dataset(images, labels):
  images = images/255.
  return images, labels

train_dataset = train_dataset.map(one_hot_in_dataset)
train_dataset = train_dataset.map(norm_in_dataset)
val_dataset = val_dataset.map(one_hot_in_dataset)
val_dataset = val_dataset.map(norm_in_dataset)
test_dataset = test_dataset.map(one_hot_in_dataset)
test_dataset = test_dataset.map(norm_in_dataset)

# train and evaluate the model
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    batch_size=batch_size
)

loss, accuracy = model.evaluate(test_dataset)

# use the model on new images
folder_path = "dataset/seg_pred/seg_pred"
file_names = os.listdir(folder_path)
preprocessed_images = []
for file_name in file_names:
  file_path = os.path.join(folder_path, file_name)
  img = image.load_img(file_path, target_size=(64,64))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = x/225.
  preprocessed_images.append(x)

preprocessed_images = np.vstack(preprocessed_images)
pred = model.predict(preprocessed_images)