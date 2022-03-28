import os
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


##################################################################################
# Load data from npz file generated in generatetraindata.py
##################################################################################
image_shape = (48, 48, 1)
labels_num = 48

# extract the first array
x_train = np.load('training_data.npz')["arr_0"]
y_train = np.load('training_labels.npz')["arr_0"]

#
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

# Normalize data to a range of 0 to 1
x_train = x_train / 255
x_test = x_test / 255

# Expand dimensions to ensure shapes are (32, 32, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, labels_num)
y_test = keras.utils.to_categorical(y_test, labels_num)

# Data augmentation to increase shift immutability
datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=15, zoom_range=0.20)
datagen.fit(x_train)

print("--SHAPES--")
print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)

##################################################################################
# Build the model
##################################################################################

model = keras.Sequential(
    [
        keras.Input(shape=image_shape),

        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.5),

        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.5),

        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(labels_num, activation="softmax"),
    ]
)

model.summary()

##################################################################################
# Train the model
##################################################################################
batch_size = 64
epochs = 14
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          callbacks=[tensorboard_callback],
          validation_data=(x_test, y_test))

##################################################################################
# Save the model
##################################################################################

for k in model.layers:
    if type(k) is keras.layers.Dropout:
        model.layers.remove(k)

model.save("SingleJPNCharOCR")
