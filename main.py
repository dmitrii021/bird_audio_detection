# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 23:45:17 2020

@author: User
"""
import numpy as np
import csv
from librosa.feature import melspectrogram
from librosa import core

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
    

train_data, train_labels = read_train_data()
# test_data = read_test_data()

    