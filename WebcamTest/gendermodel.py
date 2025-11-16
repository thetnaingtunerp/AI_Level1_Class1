from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.preprocessing import image


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop

from sklearn.model_selection import train_test_split

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

data = []
labels = []

img_dims = (96 , 96, 3)


image_files = [f for f in glob.glob(r'facedataset'+"/**/*", recursive=True) if not os.path.isdir(f)]