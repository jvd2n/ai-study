import time
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from keras.utils import np_utils

#1. Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
ic('********* raw data **********')
ic(x_train.shape, x_test.shape)
ic(y_train.shape, y_test.shape)

# Data preprocessing
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

ic('********* 1st reshape **********')
ic(x_train.shape, x_test.shape)
ic(y_train.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

ic('********* Scaling **********')
ic(x_train.shape, x_test.shape)
x_train = x_train.reshape(-1, 32 * 32, 3)
x_test = x_test.reshape(-1, 32 * 32, 3)

ic('********* 2nd reshape **********')
ic(x_train.shape, x_test.shape)


from sklearn.preprocessing import OneHotEncoder
oneEnc = OneHotEncoder()
y_train = oneEnc.fit_transform(y_train).toarray()
y_test = oneEnc.transform(y_test).toarray()

ic('********** OneHotEnc **********')
ic(y_train.shape, y_test.shape)

#2. Model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Dropout, GlobalAveragePooling1D, MaxPooling1D

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu', input_shape=(32 * 32, 3)))
model.add(Dropout(0.2))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3 Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

start_time = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=512, verbose=2, 
          validation_split=0.15, callbacks=[es])
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
ic| duration_time: 3403.9552216529846
loss: 2.045886754989624
accuracy: 0.26010000705718994

StandardScaler Conv1D
ic| duration_time: 317.79892349243164
ic| loss[0]: 1.2282058000564575
ic| loss[1]: 0.585099995136261
'''
