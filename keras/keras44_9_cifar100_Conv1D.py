import time
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from keras.utils import np_utils

#1. Data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
ic(x_train.shape, x_test.shape)

# Data preprocessing
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

ic(x_train.shape, x_test.shape)
ic(y_train.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 32 * 32, 3)
x_test = x_test.reshape(-1, 32 * 32, 3)
ic(x_train.shape, x_test.shape)

from sklearn.preprocessing import OneHotEncoder
oneEnc = OneHotEncoder()
y_train = oneEnc.fit_transform(y_train).toarray()
y_test = oneEnc.transform(y_test).toarray()

ic(x_train.shape, x_test.shape)
ic(y_train.shape, y_test.shape)

#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Dropout, GlobalAveragePooling1D, MaxPooling1D

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu', input_shape=(32 * 32, 3)))
model.add(Dropout(0.2))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3 Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

start_time = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=512, verbose=2, 
          validation_split=0.02, callbacks=[es])
duration_time = time.time() - start_time

#4 Evaluate
loss = model.evaluate(x_test, y_test)   # evaluate -> return loss, metrics

ic(duration_time)
ic(loss[0])
ic(loss[1])


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
ic| duration_time: 1253.9313232898712
ic| loss: [nan, 0.009999999776482582]

StandardScaler Conv1D
ic| duration_time: 370.3385968208313
ic| loss[0]: 3.3052778244018555
ic| loss[1]: 0.29440000653266907
'''
