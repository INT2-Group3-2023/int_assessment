import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Input, Dense, Conv2D, Add
from tensorflow.keras.layers import SeparableConv2D, ReLU
from tensorflow.keras.layers import BatchNormalization, MaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras import Model

data, ds_info = tfds.load('oxford_flowers102',
                          with_info=True,
                          as_supervised=True,
                          shuffle_files=True)
train_ds, valid_ds, test_ds = data['train'], data['validation'], data['test']


def scale_resize_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)  # equivalent to dividing image pixels by 255
    image = tf.image.resize(image, (299, 299))  # Resizing the image to 224x224 dimention
    return (image, label)


def crop_image(image, label):
    image = tf.image.resize_with_crop_or_pad(image, 500, 500)
    return (image, label)


def visualise(original, augmented):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("original")
    plt.imshow(original)

    plt.subplot(1, 2, 2)
    plt.title("augmented")
    plt.imshow(augmented)


def show_augmentations():
    image, label = next(iter(train_ds))
    # line of augmentation
    augmented = tf.image.adjust_contrast(image, 2)
    visualise(image, augmented)
    plt.show()


def plot_graph():
    plt.figure(figsize=(8, 8))
    history = open("history.txt")
    adam_acc = [float(x) for x in history.readline().strip().split(',')]
    adam_val_acc = [float(x) for x in history.readline().strip().split(',')]
    adam_loss = [float(x) for x in history.readline().strip().split(',')]
    adam_val_loss = [float(x) for x in history.readline().strip().split(',')]
    epochs_range = range((len(adam_acc)))
    plt.plot(epochs_range, adam_acc, label="Training Accuracy [Adam]")
    plt.plot(epochs_range, adam_val_acc, label="Validation Accuracy [Adam]")
    plt.plot(epochs_range, sgd_history.history['accuracy'], label="Training Accuracy [SGD]")
    plt.plot(epochs_range, sgd_history.history['val_accuracy'], label="Validation Accuracy [SGD]")
    plt.axis(ymin=0.00, ymax=1)
    plt.grid()
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train [Adam]', 'validation [Adam]', 'train [SGD]', 'validation [SGD]'])
    plt.show()
    plt.savefig('output-plot.png')

    plt.figure(figsize=(8, 8))
    epochs_range = range((len(adam_history.history['accuracy'])))
    plt.plot(epochs_range, adam_loss, label="Training Loss [Adam]")
    plt.plot(epochs_range, adam_val_loss, label="Validation Loss [Adam]")
    plt.plot(epochs_range, sgd_history.history['loss'], label="Training Loss [SGD]")
    plt.plot(epochs_range, sgd_history.history['val_loss'], label="Validation Loss [SGD]")
    plt.axis(ymin=0.00, ymax=10)
    plt.grid()
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train [Adam]', 'validation [Adam]', 'train [SGD]', 'validation [SGD]'])
    plt.show()
    plt.savefig('output-plot.png')


AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 2

# data pre-processing:

training_ds = (train_ds.map(crop_image))
training_ds = (training_ds.map(scale_resize_image))
training_ds = training_ds.batch(BATCH_SIZE)

testing_ds = (test_ds.map(crop_image))
testing_ds = (testing_ds.map(scale_resize_image))
testing_ds = testing_ds.batch(BATCH_SIZE)

validation_ds = (valid_ds.map(crop_image))
validation_ds = (validation_ds.map(scale_resize_image))
validation_ds = validation_ds.batch(BATCH_SIZE)


# data augmentation (these functions will only be active during model.fit and will deactivate for model.evaluate)
def l(x):
    x = keras.layers.RandomFlip("horizontal_and_vertical")(x)
    x = keras.layers.RandomRotation(1)(x)
    x = keras.layers.RandomZoom(0.5)(x)
    x = keras.layers.RandomBrightness(0.2 / 255)(x)
    x = keras.layers.RandomTranslation(0.1, 0.1)(x)
    x = keras.layers.BatchNormalization()(x)
    return x


input = keras.layers.Input(shape=(299, 299, 3))

tensorA = (Conv2D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False)(l(input)))
tensorA = (BatchNormalization()(tensorA))
tensorA = (ReLU()(tensorA))

tensorA = (Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False)(tensorA))
tensorA = (BatchNormalization()(tensorA))
tensorB = (ReLU()(tensorA))

tensorA = (SeparableConv2D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=False)(tensorB))
tensorA = (BatchNormalization()(tensorA))
tensorA = (ReLU()(tensorA))

tensorA = (SeparableConv2D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=False)(tensorA))
tensorA = (BatchNormalization()(tensorA))
tensorA = (MaxPool2D(pool_size=3, strides=2, padding='same')(tensorA))

tensorB = (Conv2D(filters=128, kernel_size=1, strides=2, padding='same', use_bias=False)(tensorB))
tensorB = (BatchNormalization()(tensorB))
tensorA = Add()([tensorB, tensorA])

tensorA = ReLU()(tensorA)
tensorA = SeparableConv2D(filters=256, kernel_size=3, strides=1, padding='same', use_bias=False)(tensorA)
tensorA = BatchNormalization()(tensorA)
tensorA = ReLU()(tensorA)

tensorA = SeparableConv2D(filters=256, kernel_size=3, strides=1, padding='same', use_bias=False)(tensorA)
tensorA = BatchNormalization()(tensorA)
tensorA = MaxPool2D(pool_size=3, strides=2, padding='same')(tensorA)

tensorB = Conv2D(filters=256, kernel_size=1, strides=2, padding='same', use_bias=False)(tensorB)
tensorB = BatchNormalization()(tensorB)
tensorA = Add()([tensorB, tensorA])

tensorA = ReLU()(tensorA)
tensorA = SeparableConv2D(filters=728, kernel_size=3, strides=1, padding='same', use_bias=False)(tensorA)
tensorA = BatchNormalization()(tensorA)
tensorA = ReLU()(tensorA)

tensorA = SeparableConv2D(filters=728, kernel_size=3, strides=1, padding='same', use_bias=False)(tensorA)
tensorA = BatchNormalization()(tensorA)

tensorA = MaxPool2D(pool_size=3, strides=2, padding='same')(tensorA)

tensorB = Conv2D(filters=728, kernel_size=1, strides=2, padding='same', use_bias=False)(tensorB)
tensorB = BatchNormalization()(tensorB)
tensorA = Add()([tensorB, tensorA])

tensorB = tensorA

tensorA = ReLU()(tensorB)
tensorA = SeparableConv2D(filters=728, kernel_size=3, strides=1, padding='same', use_bias=False)(tensorA)
tensorA = BatchNormalization()(tensorA)

tensorA = ReLU()(tensorA)

tensorA = SeparableConv2D(filters=728, kernel_size=3, strides=1, padding='same', use_bias=False)(tensorA)
tensorA = BatchNormalization()(tensorA)

tensorA = ReLU()(tensorA)

tensorA = SeparableConv2D(filters=728, kernel_size=3, strides=1, padding='same', use_bias=False)(tensorA)
tensorA = BatchNormalization()(tensorA)

tensorA = ReLU()(tensorA)

tensorB = Add()([tensorB, tensorA])

tensorB = tensorA
tensorA = ReLU()(tensorB)

tensorA = SeparableConv2D(filters=728, kernel_size=3, strides=1, padding='same', use_bias=False)(tensorA)
tensorA = BatchNormalization()(tensorA)

tensorA = ReLU()(tensorA)

tensorA = SeparableConv2D(filters=1024, kernel_size=3, strides=1, padding='same', use_bias=False)(tensorA)
tensorA = BatchNormalization()(tensorA)

tensorA = MaxPool2D(pool_size=3, strides=2, padding='same')(tensorA)

tensorB = Conv2D(filters=1024, kernel_size=1, strides=2, padding='same', use_bias=False)(tensorB)
tensorB = BatchNormalization()(tensorB)

tensorA = Add()([tensorB, tensorA])

tensorA = SeparableConv2D(filters=1536, kernel_size=3, strides=1, padding='same', use_bias=False)(tensorA)
tensorA = BatchNormalization()(tensorA)

tensorA = ReLU()(tensorA)

tensorA = SeparableConv2D(filters=2048, kernel_size=3, strides=1, padding='same', use_bias=False)(tensorA)
tensorA = BatchNormalization()(tensorA)

tensorA = GlobalAvgPool2D()(tensorA)

tensorA = Dense(units=102, activation='softmax')(tensorA)
output = tensorA

checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=20, min_lr=0.00001)

# model = keras.models.load_model('/workspace/model1')

adam_model = Model(inputs=input, outputs=output)
adam_model.summary()
sgd_model = Model(inputs=input, outputs=output)
sgd_model.summary()

adam_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

adam_history = adam_model.fit(training_ds, validation_data=validation_ds, epochs=5, verbose=2,
                              callbacks=[model_checkpoint_callback, reduce_lr])

# write results to file before clearing the session for SGD
file = open('history.txt', 'w')
file.writelines(','.join(map(str, adam_history.history['accuracy'])))
file.writelines('\n')
file.writelines(','.join(map(str, adam_history.history['val_accuracy'])))
file.writelines('\n')
file.writelines(','.join(map(str, adam_history.history['loss'])))
file.writelines('\n')
file.writelines(','.join(map(str, adam_history.history['val_loss'])))
file.close()

keras.backend.clear_session()

sgd_model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

sgd_history = sgd_model.fit(training_ds, validation_data=validation_ds, epochs=5, verbose=2,
                            callbacks=[model_checkpoint_callback, reduce_lr])

plot_graph()

adam_model.evaluate(testing_ds)
sgd_model.evaluate(testing_ds)

# model.save('/workspace/savedmodels')
