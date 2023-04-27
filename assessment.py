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

#transform images to the chosen dimension (they are just squeezed)
def scale_resize_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) # equivalent to dividing image pixels by 255
    image = tf.image.resize(image, (120, 120)) # Resizing the image to 224x224 dimention
    return (image, label)

#converts images to greyscale
def rgb_convert(image, label):
    image = tf.image.rgb_to_grayscale(image)
    return (image, label)

#crops image about the centre
def crop_image(image, label):
    image = tf.image.resize_with_crop_or_pad(image, 500, 500)
    return (image, label)

AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 256

training_ds = (train_ds.map(crop_image))
training_ds = (training_ds.map(scale_resize_image))
training_ds = (training_ds.map(rgb_convert))
training_ds = training_ds.batch(BATCH_SIZE)

testing_ds = (test_ds.map(scale_resize_image))
testing_ds = testing_ds.map(rgb_convert)
testing_ds = testing_ds.batch(BATCH_SIZE)

#for example in training_ds:
#  print(tf.shape(tf.expand_dims(example[0], axis = 0)))

#display the resized images
#fig = tfds.show_examples(training_ds, ds_info, rows = 4, cols = 4)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(120,120,1)),
    keras.layers.Dense(102, activation=tf.nn.softmax)
])

model.summary()

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 0.001),
    loss = keras.losses.SparseCategoricalCrossentropy(),
    metrics = ["accuracy"]
)

model.fit(training_ds, epochs = 5, verbose=2)
model.evaluate(testing_ds)