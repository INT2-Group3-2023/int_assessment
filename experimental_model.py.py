# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Input,Dense,Conv2D,Add
from tensorflow.keras.layers import SeparableConv2D,ReLU
from tensorflow.keras.layers import BatchNormalization,MaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras import Model


def conv_bn(x, filters, kernel_size, strides=1):
  x = Conv2D(filters=filters, 
             kernel_size = kernel_size, 
             strides=strides, 
             padding = 'same', 
             use_bias = False)(x)
  x = BatchNormalization()(x)
  return x

# creating separableConv-Batch Norm block

def sep_bn(x, filters, kernel_size, strides=1):
  x = SeparableConv2D(filters=filters, 
                      kernel_size = kernel_size, 
                      strides=strides, 
                      padding = 'same', 
                      use_bias = False)(x)
  x = BatchNormalization()(x)
  return x

input = keras.layers.Input(shape = (299,299,3))
#x = keras.layers.RandomFlip("horizontal_and_vertical")(input)
#x = keras.layers.RandomRotation(1)(x)
#x = keras.layers.RandomZoom(0.5)(x)
#x = keras.layers.RandomBrightness(0.2/255)(x)
#x = keras.layers.RandomTranslation(0.1, 0.1)(x)
#x = keras.layers.BatchNormalization()(x)  
x = conv_bn(input, filters =32, kernel_size =3, strides=2)
x = ReLU()(x)
x = conv_bn(x, filters =64, kernel_size =3, strides=1)
tensor = ReLU()(x)
    
x = sep_bn(tensor, filters = 128, kernel_size =3)
x = ReLU()(x)
x = sep_bn(x, filters = 128, kernel_size =3)
x = MaxPool2D(pool_size=3, strides=2, padding = 'same')(x)
    
tensor = conv_bn(tensor, filters=128, kernel_size = 1,strides=2)
x = Add()([tensor,x])
    
x = ReLU()(x)
x = sep_bn(x, filters =256, kernel_size=3)
x = ReLU()(x)
x = sep_bn(x, filters =256, kernel_size=3)
x = MaxPool2D(pool_size=3, strides=2, padding = 'same')(x)
    
tensor = conv_bn(tensor, filters=256, kernel_size = 1,strides=2)
x = Add()([tensor,x])
    
x = ReLU()(x)
x = sep_bn(x, filters =728, kernel_size=3)
x = ReLU()(x)
x = sep_bn(x, filters =728, kernel_size=3)
x = MaxPool2D(pool_size=3, strides=2, padding = 'same')(x)
    
tensor = conv_bn(tensor, filters=728, kernel_size = 1,strides=2)
x = Add()([tensor,x])

tensor = x
for _ in range(8):
    x = ReLU()(tensor)
    x = sep_bn(x, filters = 728, kernel_size = 3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 728, kernel_size = 3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 728, kernel_size = 3)
    x = ReLU()(x)
    tensor = Add()([tensor,x])

tensor = x
x = ReLU()(tensor)
x = sep_bn(x, filters = 728,  kernel_size=3)
x = ReLU()(x)
x = sep_bn(x, filters = 1024,  kernel_size=3)
x = MaxPool2D(pool_size = 3, strides = 2, padding ='same')(x)
    
tensor = conv_bn(tensor, filters =1024, kernel_size=1, strides =2)
x = Add()([tensor,x])

x = sep_bn(x, filters = 1536,  kernel_size=3)
x = ReLU()(x)
x = sep_bn(x, filters = 2048,  kernel_size=3)
x = GlobalAvgPool2D()(x)
    
x = Dense(units = 102, activation = 'softmax')(x)
output = x

model = Model (inputs=input, outputs=output)
model.summary()


data, ds_info = tfds.load('oxford_flowers102',
                         with_info=True,
                          as_supervised=True,
                          shuffle_files = True)
train_ds, valid_ds, test_ds = data['train'], data['validation'], data['test']

def normalize_image(image, label):
    image = tf.image.resize_with_crop_or_pad(image, 500, 500)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (299, 299))
    return (image, label)


AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 8

training_ds = (train_ds.map(normalize_image)).cache().batch(BATCH_SIZE)

testing_ds = (test_ds.map(normalize_image)).cache().batch(BATCH_SIZE)

validation_ds = (valid_ds.map(normalize_image)).cache().batch(BATCH_SIZE)

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 0.001),
    loss = keras.losses.SparseCategoricalCrossentropy(),
    metrics = ["accuracy"]
)

checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


model.fit(training_ds, validation_data = validation_ds, epochs = 700, verbose=2, callbacks = [model_checkpoint_callback])

model.evaluate(testing_ds)

