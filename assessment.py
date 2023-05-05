#!/usr/bin/env python
# coding: utf-8

# In[12]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data, ds_info = tfds.load('oxford_flowers102',
                         with_info=True,
                          as_supervised=True,
                          shuffle_files = True)
train_ds, valid_ds, test_ds = data['train'], data['validation'], data['test']

type(train_ds)

print(type(train_ds))

def scale_resize_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) # equivalent to dividing image pixels by 255
    image = tf.image.resize(image, (224, 224)) # Resizing the image to 224x224 dimention
    return (image, label)

def rgb_convert(image, label):
    image = tf.image.rgb_to_grayscale(image)
    return (image, label)

def crop_image(image, label):
    image = tf.image.resize_with_crop_or_pad(image, 500, 500)
    return (image, label)

def resize_with_crop_or_pad(image, label):
    image = tf.image.resize_with_crop_or_pad(image, 752, 752)
    return (image, label)

AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128

#training_ds = (train_ds.map(crop_image))
training_ds = (train_ds.map(scale_resize_image))
#training_ds = (training_ds.map(rgb_convert))
#training_ds = train_ds.map(resize_with_crop_or_pad)
training_ds = training_ds.batch(BATCH_SIZE)

#testing_ds = (test_ds.map(crop_image))
testing_ds = (test_ds.map(scale_resize_image))
#testing_ds = testing_ds.map(rgb_convert)
#testing_ds = test_ds.map(resize_with_crop_or_pad)
testing_ds = testing_ds.batch(BATCH_SIZE)

counter = 0
for example in train_ds:
  counter = counter + 1
print(counter)

def visualise(original, augmented):
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.title("original")
    plt.imshow(original)

    plt.subplot(1,2,2)
    plt.title("augmented")
    plt.imshow(augmented)


def show_augmentations():
    image, label = next(iter(train_ds))
    # line of augmentation
    augmented = tf.image.resize_with_crop_or_pad(image, 752, 752)
    visualise(image, augmented)
    plt.show()

show_augmentations()

model = keras.Sequential([
    #keras.layers.CenterCrop(500, 500),
    #keras.layers.Resizing(128, 128),
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(1),
    keras.layers.RandomZoom(0.5),
    #keras.layers.RandomContrast(1),
    #keras.layers.RandomBrightness(0.2/255),
    # line of testing!
    keras.Input((128, 128, 3)),
    keras.layers.Conv2D(4, 3, activation='relu'),
    #keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    
    keras.layers.Conv2D(8, 3, activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    #keras.layers.Dropout(0.2),
    
    keras.layers.Conv2D(16, 3, activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    #keras.layers.Dropout(0.2),
    
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    #keras.layers.Dropout(0.2),
    
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    #keras.layers.Dropout(0.2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(102, activation=tf.nn.softmax)
])

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 0.001),
    loss = keras.losses.SparseCategoricalCrossentropy(),
    metrics = ["accuracy"]
)

checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='accuracy',
    mode='max',
    save_best_only=True)

epochs = 300

history = model.fit(training_ds, epochs = epochs, verbose=2, callbacks = [model_checkpoint_callback])
model.load_weights(checkpoint_filepath)

plt.figure(figsize=(8, 8))
epochs_range= range(epochs)
plt.plot( epochs_range, history.history['accuracy'], label="Training Accuracy")
#plt.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")
plt.axis(ymin=0.85,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()
plt.savefig('output-plot.png')

model.evaluate(testing_ds)
