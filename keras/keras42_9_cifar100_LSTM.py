import time
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from keras.utils import np_utils

#1. Data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Data preprocessing
x_train = x_train.reshape(-1, 32 * 32 * 3)
x_test = x_test.reshape(-1, 32 * 32 * 3)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(-1, 32 * 32, 3)
x_test = x_test.reshape(-1, 32 * 32, 3)

ic(x_train.shape, x_test.shape)
ic(y_train.shape, y_test.shape)

from sklearn.preprocessing import OneHotEncoder
oneEnc = OneHotEncoder()
y_train = oneEnc.fit_transform(y_train).toarray()
y_test = oneEnc.transform(y_test).toarray()

#2. Model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling1D, LSTM, Input

model = Sequential()
input1 = Input(shape=(32 * 32, 3))
xx = LSTM(units=10, activation='relu')(input1)
xx = Dense(256, activation='relu')(xx)
xx = Dense(128, activation='relu')(xx)
output1 = Dense(100, activation='softmax')(xx)
model = Model(inputs=input1, outputs=output1)

#3 Compile, Train   metrics=['accuracy']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

start_time = time.time()
model.fit(x_train, y_train, epochs=20, batch_size=512, 
          validation_split=0.1)
duration_time = time.time() - start_time

#4 Evaluate
loss = model.evaluate(x_test, y_test)   # evaluate -> return loss, metrics

ic(duration_time)
ic(loss)


'''
CNN
loss: 0.05057989060878754
accuracy: 0.9922999739646912

DNN
loss: 0.17536625266075134
accuracy: 0.9753999710083008

DNN + GAP
loss: 1.7715743780136108
accuracy: 0.35740000009536743

LSTM

'''
