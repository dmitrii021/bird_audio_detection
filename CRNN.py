#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 23:35:29 2020

@author: ben
"""
import numpy as np
import csv
import keras
from librosa.feature import melspectrogram
from librosa import core
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, Conv1D, Flatten, Bidirectional, LSTM, Dropout, BatchNormalization, TimeDistributed
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Reshape

# Fix random seed for reproducability
from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)

#%%
train_data = np.load('/content/drive/MyDrive/bird_detection/train_data.npy')
train_labels = np.load('/content/drive/MyDrive/bird_detection/train_labels.npy')
test_data = np.load('/content/drive/MyDrive/bird_detection/test_data.npy')
train_labels = np.asarray(train_labels, dtype = int)


#%%
train_data, X_valid, train_labels, y_valid = train_test_split(train_data, train_labels, test_size = 0.2)



#%%
opt = keras.optimizers.Adam(learning_rate=0.002)
model = Sequential()
#
model.add(TimeDistributed(Conv1D(96, 5, padding='same', activation='relu'), input_shape = (862,40, 1)))
model.add(TimeDistributed(MaxPooling1D()))
model.add(TimeDistributed(Conv1D(96, 5, padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling1D()))
model.add(TimeDistributed(Conv1D(96, 5, padding='same', activation='relu')))


model.add(Reshape((862, 1920)))
model.add(Bidirectional(LSTM(80, return_sequences=True)))

model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer = opt, metrics = ['accuracy'])


#%%
history = model.fit(np.expand_dims(train_data, 3), train_labels, batch_size=32, epochs = 9, validation_data = (np.expand_dims(X_valid, 3), y_valid))
probs = model.predict(np.expand_dims(test_data, 3), verbose=1)

model.summary()

# Write to csv file
with open("submission1.csv", "w") as fp: 
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