# -*- coding: utf-8 -*-
"""
Code for Preprocessing data for Deep Learning Model for Plant Pathology 2020 - FGVC7


Author : JN

"""

import numpy as np
import pandas as pd
print(tf.__version__)
import os
import shutil
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras_preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Load image data
def load_data(path, size):
    train_images = np.ndarray(shape=(train_len, size, size, 3))
    for i in tqdm(range(train_len)):
        img = load_img(path + f'Train_{i}.jpg', target_size=(size, size))
        train_images[i] = np.uint8(img_to_array(img))

    test_images = np.ndarray(shape=(test_len, size, size, 3))
    for i in tqdm(range(test_len)):
        img = load_img(path + f'Test_{i}.jpg', target_size=(size, size))
        test_images[i] = np.uint8(img_to_array(img))
        
# Split data
def split_data(test_size, random_state):
    x_train, x_test, y_train, y_test = train_test_split(train_images, target.to_numpy(), test_size=test_size, random_state=random_state) 
    return x_train, x_test, y_train, y_test

# Random Over Sampling to fix class imbalances
def oversample(random_state):
    ros = RandomOverSampler(random_state=random_state)
    x_train, y_train = ros.fit_resample(x_train.reshape((-1, size * size * 3)), y_train)
    x_train = x_train.reshape((-1, size, size, 3))
    return x_train, y_train


def get_generators(batch_size)
    train_datagen = ImageDataGenerator(samplewise_center = True,
                                   samplewise_std_normalization = True,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   rotation_range=20)

    train_generator = train_datagen.flow(
    x = x_train, 
    y = y_train,
    batch_size = batch_size)

    validation_datagen = ImageDataGenerator(samplewise_center = True,
                                        samplewise_std_normalization = True)

    validation_generator = validation_datagen.flow(
    x = x_test, 
    y = y_test,
    batch_size = batch_size)
    
    return train_generator, valid_generator


