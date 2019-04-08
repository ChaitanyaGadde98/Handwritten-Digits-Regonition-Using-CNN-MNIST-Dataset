import numpy as np
import cv2

width, height = 28, 28

image = cv2.imread('test/img_1.jpg')
image = cv2.resize(image, (width, height))
