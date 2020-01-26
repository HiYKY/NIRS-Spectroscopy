# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 19:57:57 2019

@author: Prabahar Ravichandran
"""

import numpy as np
import pandas as pd
import os
import tensorflow as tf
import keras_radam

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from keras import models, layers, optimizers, datasets, utils, losses
from keras.utils import to_categorical
from keras import regularizers, optimizers
from keras.models import load_model

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

FilePath=r'C:/Users/Praba/Desktop/5_NIRS Amylose/Phase 1 Study files/FOSS_NIRS_Amylose/Data_only.csv'

File=pd.read_csv(FilePath, header=None)
FileNumpy = File.to_numpy()
Amylose = list(FileNumpy[1, 1:])
Entry = list(FileNumpy[0, 1:])
Data = np.delete(FileNumpy, [0], axis = 0)
Data = np.delete(Data, [0], axis = 0)
#print(Data[0])
wl = list(Data[:,0])
print(len(wl))
Data = np.delete(Data, [0], axis = 1)
data_list = []


for i in range(int((np.size(Data,1)))):
    arr = np.resize(Data[:,i], 2245)
    data_list.append(arr)

data_array = np.asarray(data_list)
data_array = data_array.astype('float32')
print(data_array.shape)

x_train = data_array
print(x_train[0])
print(x_train[0].shape)

y_train = np.transpose(np.matrix(Amylose))
print(y_train[0])
print(y_train[0].shape)



X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.1)

## More Layers

model = models.Sequential()
model.add(layers.Dense(256, input_dim = 2245, activation='relu'))
model.add(layers.Dense(164, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(36, activation='relu'))
model.add(layers.Dense(1, activation='relu'))

#aadam = keras_radam.RAdam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
#ssgd = optimizers.SGD(lr=0.00001, decay=1e-6, momentum = 0.75, nesterov = True)
model.compile(loss='mean_squared_error',
              optimizer='Adam',
              metrics=['mean_squared_error'])

history = model.fit(X_train, Y_train, validation_split=0.1, batch_size = 10, epochs=500, shuffle = True, verbose=1)
model.summary()
model.save('C:/Users/Praba/Desktop/Weights/model_0.h5')
predicted_data = np.asarray(model.predict(x_train))
actual_data = np.asarray(y_train.astype('float32'))
R2 = r2_score(actual_data, predicted_data) 
print(R2)
#data = np.concatenate((actual_data, predicted_data), axis=1)
#np.savetxt('C:/Users/Praba/Desktop/Output.csv', data, fmt='%1.2f', delimiter = ',')

plt.plot(history.history['mean_squared_error'])
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.plot(history.history['val_mean_squared_error'])
plt.xlabel('Epochs')
plt.ylabel('MSE')


i = 25
R2_list=[]
for j in range(i):
    model_in = 'C:/Users/Praba/Desktop/Weights/'+ 'model_'+ str(j+0) + '.h5'
    model_out = 'C:/Users/Praba/Desktop/Weights/'+ 'model_'+ str(j+1) + '.h5'    
    model_load = load_model(model_in)
    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.1)
    aadam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
    model.compile(loss='mean_squared_error',
              optimizer=aadam,
              metrics=['mean_squared_error'])
    history = model.fit(X_train, Y_train, validation_split=0.1, batch_size = 10, epochs=100, shuffle = True, verbose=1)
    model.summary()
    model_load.save(model_out)  
    predicted_data = np.asarray(model.predict(x_train))
    actual_data = np.asarray(y_train.astype('float32'))
    R2 = r2_score(actual_data, predicted_data) 
    print(R2)
    data = np.concatenate((actual_data, predicted_data), axis=1)
    np.savetxt('C:/Users/Praba/Desktop/Output.csv', data, fmt='%1.2f', delimiter = ',')
    
    plt.plot(history.history['mean_squared_error'])
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.plot(history.history['val_mean_squared_error'])
    plt.xlabel('Epochs')
    plt.ylabel('MSE')

