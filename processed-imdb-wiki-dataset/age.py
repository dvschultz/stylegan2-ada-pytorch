# Importing dependencies
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# Loading dataset
meta = pd.read_csv('meta.csv')

# Filtaring dataset
meta = meta[meta['age'] >= 0]
meta = meta[meta['age'] <= 9]

# Converting into numpy array
meta = meta.values

# Making the directory structure
for i in range(10):
    output_dir = 'dataset/' + str(i)

# Finally making the training and testing set
counter = 0

for image in meta:
    img = cv2.imread(image[1], 1)
    img = cv2.resize(img, (128, 128))
    cv2.imwrite('dataset/' + str(image[0]) + '/' + str(counter) + '.jpg', img)
    print('--('+str(counter)+')Processing--')
    counter += 1


