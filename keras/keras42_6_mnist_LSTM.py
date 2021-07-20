import time
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils

#1. Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
ic(x_train.shape, x_test.shape)

# Data preprocessing
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2], 1)

ic(x_train.shape, x_test.shape)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling1D, LSTM

model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(x_train.shape[1] * x_train.shape[2], 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3 Compile, Train   metrics=['accuracy']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

start_time = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=2,
    validation_split=0.02, callbacks=[es])
end_time = time.time()
duration_time = end_time - start_time

#4 Evaluate
ic('================= EVALUATE ==================')
loss = model.evaluate(x_test, y_test)   # evaluate -> return loss, metrics
ic(duration_time)
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
