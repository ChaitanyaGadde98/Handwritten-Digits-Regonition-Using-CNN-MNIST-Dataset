import numpy as np
import cv2
from keras.models import load_model
import csv

width, height = 28, 28

model = load_model('mnist_cnn.model')

image = cv2.imread('test/img_1.jpg')
    
def img_process(image):
    image = cv2.resize(image, (width, height))
    img = np.array(image).reshape((width, height, 3))
    img = np.expand_dims(img, axis = 0)
    return img

def prediction(img):
    prediction = model.predict(img)[0]
    for i in range(10):
        if(prediction[i] == 1):
            return i

print(prediction(img_process(image)))

