import time
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

#1. Data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Data preprocessing
ic(x_train.shape, x_test.shape)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

ic(x_train.shape, x_test.shape)
x_train = x_train.reshape(-1, 28 * 28, 1)
x_test = x_test.reshape(-1, 28 * 28, 1)

ic(x_train.shape, x_test.shape)

from sklearn.preprocessing import OneHotEncoder
oneEnc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = oneEnc.fit_transform(y_train).toarray()
y_test = oneEnc.transform(y_test).toarray()
ic(y_train.shape, y_test.shape)

#2. Model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Flatten, Dropout, GlobalAveragePooling1D, MaxPooling1D

model = Sequential()
input1 = Input(shape=(28*28, 1))
xx = LSTM(units=10, activation='relu')(input1)
xx = Dense(128, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
xx = Dense(16, activation='relu')(xx)
output1 = Dense(10, activation='softmax')(xx)
model = Model(inputs=input1, outputs=output1)

#3 Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

start_time = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=512, verbose=2, 
          validation_split=0.02, callbacks=[es])
duration_time = time.time() - start_time


#4 Evaluate
ic('================= EVALUATE ==================')
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
ic| duration_time: 461.7817986011505
ic| loss[0]: 1.5526938438415527
ic| loss[1]: 0.37720000743865967
'''
