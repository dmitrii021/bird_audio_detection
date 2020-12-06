#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:16:18 2020

@author: ben
"""
import numpy as np
import csv
import keras
from librosa.feature import melspectrogram
from librosa import core
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, Flatten, Bidirectional, LSTM, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from skimage.util import random_noise


# Load data 
train_data = np.load('/content/drive/MyDrive/bird_detection/train_data.npy')
train_labels = np.load('/content/drive/MyDrive/bird_detection/train_labels.npy')
test_data = np.load('/content/drive/MyDrive/bird_detection/test_data.npy')
train_labels = np.asarray(train_labels, dtype = int)

# Adding Gaussian Noise to data
noisy_X = np.zeros(train_data.shape)
for n in range(train_data.shape[0]):
    noisy_X[n, ...] = random_noise(train_data[n, ...], mode='gaussian', mean=0, var=5)

np.save("noisy_train_data.npy", noisy_X)

#%%


def data_loader(data_files, label_files, batch_size):
    while True:

        for n in range(len(data_files)):
          data = np.load(data_files[n])
          labels = np.load(label_files[n])
          labels = np.asarray(labels, dtype = int)

          batch_start = 0
          batch_end = batch_size
          L = labels.shape[0]
          while batch_start < L:
              limit = min(batch_end, L)
              X = data[batch_start:limit]
              Y = labels[batch_start:limit]

              yield (X,Y)

              batch_start += batch_size
              batch_end += batch_size
              
data_files = ["train_data.npy", "noisy_train_data.npy"]
label_files = ["train_labels.npy", "train_labels.npy"]


#%%
opt = keras.optimizers.Adam(learning_rate=0.002)
model = Sequential()
model.add(BatchNormalization(input_shape = (862, 40)))
model.add(Bidirectional(LSTM(80, return_sequences=False, input_shape=(862, 40))))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
model.summary()


#%%
history = model.fit(data_loader(data_files, label_files, 80), epochs=1, use_multiprocessing=False)

test_data = np.load('/content/drive/MyDrive/bird_detection/test_data.npy')

probs = model.predict(test_data, verbose=1)

#Write to csv file
with open("submission.csv", "w") as fp: 
    fp.write("ID,Predicted\n") 
    for idx in range(test_data.shape[0]): 
        fp.write(f"{idx:05},{probs[idx][0]}\n") 

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()










