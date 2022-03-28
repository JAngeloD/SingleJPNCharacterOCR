import numpy as np
import cv2
from tensorflow import keras
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##################################################################################
# Load model
##################################################################################

model = keras.models.load_model("SingleJPNCharOCR")

##################################################################################
# Predict on a single image
##################################################################################

for i in range(1, 9):
    image_name = "testcharacter"
    image_num = str(i)
    image = Image.open(image_name + image_num + '.png').convert('L')
    image = PIL.ImageOps.invert(image)
    print("Showing test character: ", image_num)

    # Retrieves pixel information and stores it into a ndarray
    image_data = image.getdata()
    image_matrix = np.array(image_data).reshape(63, 64).astype("float32")
    image_matrix = cv2.resize(image_matrix, dsize=(48, 48)) / 255

    plt.imshow(image_matrix, cmap='gray')
    plt.show()

    x_predict = np.zeros([1, 48, 48])
    x_predict[0] = image_matrix
    x_predict = np.expand_dims(x_predict, -1)

    prediction = model.predict(x_predict, verbose=1)

    print(prediction)
