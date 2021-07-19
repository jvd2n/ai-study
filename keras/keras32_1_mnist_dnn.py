from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils

#1. Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data preprocessing
x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)
# x_train = x_train.reshape(60000, 28 * 28, 1)
# x_test = x_test.reshape(10000, 28 * 28, 1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, PowerTransformer
# scaler = MinMaxScaler()
# scaler = PowerTransformer()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling1D

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(28 * 28,)))
# model.add(Dense(100, activation='relu', input_shape=(28 * 28, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
# model.add(GlobalAveragePooling1D())
model.add(Dense(10, activation='softmax'))

# DNN 구해서 CNN 비교
# DNN + GlobalAveragePooling 구해서 CNN 비교

#3 Compile, Train   metrics=['accuracy']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=2,
    validation_split=0.00001, callbacks=[es])


#4 Evaluate
ic('================= EVALUATE ==================')
loss = model.evaluate(x_test, y_test)   # evaluate -> return loss, metrics
print(f'loss: {loss[0]}')
print(f'accuracy: {loss[1]}')


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
'''
