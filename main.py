# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 23:45:17 2020

@author: User
"""
import numpy as np
import csv
import keras
from librosa.feature import melspectrogram
from librosa import core
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, Flatten, Bidirectional, LSTM, Dropout, BatchNormalization
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.preprocessing import image

def read_train_data():
    folder1_size = 7690
    folder2_size = 7990
    labels = []
    spectrograms = np.empty((folder1_size + folder2_size, 862, 40))
    i = 0
    with open('warblrb10k_public_metadata_2018.csv', mode ='r') as csv_file:
        datareader = csv.DictReader(csv_file)
        for row in datareader:
            labels.append(row['hasbird'])
            audio, sr = core.load('wav2/'+row['itemid']+'.wav', sr = 44100)
            audio = audio * 1/np.max(np.abs(audio))
            x = melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=512, n_mels=40)
            x = x.T
            if len(x) < 862:
                x = np.concatenate((x, np.zeros((862-len(x), 40))), axis = 0)
            spectrograms[i,:,:] = x[0:862, :]
            i+=1
    print(str(i))        
    with open('ff1010bird_metadata_2018.csv', mode ='r') as csv_file:
        datareader = csv.DictReader(csv_file)
        for row in datareader:
            labels.append(row['hasbird'])
            audio, sr = core.load('wav/'+row['itemid']+'.wav', sr=44100)
            audio = audio * 1/np.max(np.abs(audio))
            x = melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=512, n_mels=40)
            x = x.T
            spectrograms[i,:,:] = x[0:862, :]
            i+=1
    print(str(i))

            
    return spectrograms, labels

def read_test_data():
    data = np.empty((4512, 862, 40))
    for i in np.arange(4512):
        audio = np.load('audio/'+str(i)+'.npy')
        # Resampling to 44100
        audio = core.resample(audio, orig_sr = 48000, target_sr = 44100)
        audio = audio * 1/np.max(np.abs(audio)) 
        spec = melspectrogram(y = audio, sr = 44100, n_fft = 1024, hop_length = 512, n_mels = 40)
        spec = spec.T
        data[i, :, :] = spec
        
    return data
    
#%%
train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')
# test_data = np.load('test_data.npy')
train_labels = np.asarray(train_labels, dtype = int)




#%%
X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_labels, test_size = 0.2)
del train_data
del train_labels
#%%
y_train = keras.utils.to_categorical(y_train);
y_valid = keras.utils.to_categorical(y_valid);
#%%
ratio = 0.3
X_train = X_valid[0:int(ratio*len(X_train)), :, :]
X_valid = X_valid[0:int(ratio*len(X_valid)), :, :]
y_train = y_train[0:int(ratio*len(y_train)), :]
y_valid = y_valid[0:int(ratio*len(y_valid)), :]


#%%
opt = keras.optimizers.Adam(learning_rate=0.002)
model = Sequential()
# 
model.add(BatchNormalization(input_shape = (862, 40)))
model.add(Bidirectional(LSTM(80, return_sequences=False), input_shape=(862, 40)))
# model.add(Attention(862))
#model.add(Dropout(0.2))
model.add(Dense(200, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
model.summary()

#%%
model.fit(X_train,y_train, batch_size=32, epochs = 20, validation_data = (X_valid, y_valid))
probs = model.predict(test_data, verbose=1)
