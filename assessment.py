import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

data, ds_info = tfds.load('oxford_flowers102', 
                         with_info=True, 
                          as_supervised=True)
train_ds, valid_ds, test_ds = data['train'], data['validation'], data['test']

type(train_ds)

print(type(train_ds))

#transform images
def scale_resize_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) # equivalent to dividing image pixels by 255
    image = tf.image.resize(image, (120, 120)) # Resizing the image to 120x120 dimension
    return (image, label)

AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 256

training_ds = (
    train_ds
    .map(scale_resize_image)
)

training_ds = training_ds.batch(BATCH_SIZE)

for example in training_ds:
  print(tf.shape(tf.expand_dims(example[0], axis = 0)))

model = keras.Sequential([
    keras.Input((120, 120, 3)),
    layers.Conv2D(100, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(102),
])

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 0.001),
    loss = keras.losses.SparseCategoricalCrossentropy(),
    metrics = ["accuracy"]
)

model.fit(training_ds, epochs = 5, verbose=2)