import cv2
import os
import numpy as np


path = os.getcwd() + '\\unaugmented_data\\'
new_dir = os.getcwd() + '\\augmented_data\\'

for n, i in enumerate(os.listdir(path)):
    filename = os.fsdecode(i)
    if filename.endswith('.png'): 
        print(f'{n + 1}: Augmenting {filename}')

        img = cv2.imread(path + filename)

        cv2.imwrite(new_dir + '\\' + filename, img)

        for j in range(0,3):
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(new_dir + '\\rotated_' + str(j) + '_' + filename, img)

        for j in range(0,4):
            img = np.where((255 - 5) < 5,255,img+5)
            cv2.imwrite(new_dir + '\\brightened_' + str(j) + '_' + filename, img)

        print(f'{n + 1}: Done augmenting {filename}')