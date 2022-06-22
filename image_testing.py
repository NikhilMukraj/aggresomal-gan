import cv2
import os
import numpy as np


path = os.getcwd()
img = cv2.imread(path + '\\unaugmented_data\\0d0f30d8-bb9e-11e8-b2b9-ac1f6b6435d0_1_blue.png', 0)

cv2.imshow('img', img)
cv2.waitKey(0)

for i in range(0,3):
    img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('img', img)
    cv2.waitKey(0)


for i in range(0,4):
    img = np.where((255 - 5) < 5,255,img+5)
    cv2.imshow('img', img)
    cv2.waitKey(0)