from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from keras.utils import np_utils

#1. Data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(28 * 28,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3 Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=2,
    validation_split=0.025, callbacks=[es])


#4 Evaluate
ic('================= EVALUATE ==================')
loss = model.evaluate(x_test, y_test)   # evaluate -> return loss, metrics
print(f'loss: {loss[0]}')
print(f'acc: {loss[1]}')

'''
CNN
loss: 0.7581470012664795
accuracy: 0.9103999733924866
DNN
loss: 0.37892159819602966
accuracy: 0.8755000233650208
'''