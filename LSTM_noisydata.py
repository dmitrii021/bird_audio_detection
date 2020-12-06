#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:16:18 2020

@author: ben
"""
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, Flatten, Bidirectional, LSTM, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from skimage.util import random_noise


# Load data 
train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')
test_data = np.load('test_data.npy')
train_labels = np.asarray(train_labels, dtype = int)

# Adding Gaussian Noise to data
noisy_X = np.zeros(train_data.shape)
for n in range(train_data.shape[0]):
    noisy_X[n, ...] = random_noise(train_data[n, ...], mode='gaussian', mean=0, var=0.5)

train_data = np.concatenate(train_data, noisy_X)
train_labels = np.concatenate(train_labels, train_labels)

train_data, X_valid, train_labels, y_valid = train_test_split(train_data, train_labels, test_size = 0.2)

#%%
opt = keras.optimizers.Adam(learning_rate=0.002)
model = Sequential()
model.add(BatchNormalization(input_shape = (862, 40)))
model.add(Bidirectional(LSTM(80, return_sequences=False, input_shape=(862, 40))))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
model.summary()

history = model.fit(train_data, train_labels, batch_size=32, epochs = 2, validation_data = (X_valid, y_valid))

#%%

del train_data
del train_labels
del X_valid
del y_valid

test_data = np.load('test_data.npy')

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










