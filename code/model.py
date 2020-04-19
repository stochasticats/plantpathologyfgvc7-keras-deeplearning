# -*- coding: utf-8 -*-
"""
Deep Learning Model code for Plant Pathology 2020 - FGVC7

@author: JN
"""

import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.__version__)
import os
import shutil
import matplotlib.pyplot as plt

def create_model():
    pre_trained = tf.keras.applications.MobileNet(input_shape=(224, 224, 3), weights='imagenet', include_top=False, dropout = 0.3)
    for layer in pre_trained.layers:
      layer.trainable = False
    
    model = tf.keras.Sequential([
      pre_trained,
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(4, activation='softmax')
      ])
    model.compile(
        loss = 'kullback_leibler_divergence', 
        optimizer = 'adam', 
        metrics = ['accuracy'])
    return model

def create_callbacks(start_lr = 0.00001, min_lr = 0.00001, max_lr = 0.00005, rampup_epochs = 15, sustain_epochs = 10, exp_decay = .8):
    es = tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=1)
    mc = tf.keras.callbacks.ModelCheckpoint('model.hdf5', save_best_only=True, verbose=0)
    rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1)

    def lrfn(epoch):
        if epoch < rampup_epochs:
            return (max_lr - start_lr)/rampup_epochs * epoch + start_lr
        elif epoch < rampup_epochs + sustain_epochs:
            return max_lr
        else:
            return min_lr
    
    lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch),, verbose=True)
    return es, mc, rlr, lr

def run_model():
    history = model.fit(
                x = train_generator,  
                validation_data = validation_generator,
                epochs = epochs,
                steps_per_epoch = steps_per_epoch,
                validation_steps = validation_steps,
                verbose=1,
                callbacks=[es, lr, mc, rlr])
    return history