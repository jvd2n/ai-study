from icecream import ic
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils

#1 Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

ic(y_train[5:])
y_train = y_train.reshape(60000, 1)
y_test = y_test.reshape(10000, 1)

from sklearn import preprocessing
one_enc = preprocessing.OneHotEncoder()
y_train = one_enc.fit_transform(y_train).toarray()
y_test = one_enc.transform(y_test).toarray()
ic(y_train.shape, y_test.shape)
ic(y_train[5:])
# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#2 Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Reshape, Dropout

model = Sequential()
# model.add(Conv2D(filters=100, kernel_size=(2,2), padding='same', input_shape=(28,28,1)))
model.add(Dense(units=10, activation='relu', input_shape=(28, 28)))
model.add(Flatten())    # (N, 280)
model.add(Dense(784))   # (N, 784)
model.add(Reshape((28, 28, 1)))     # (N, 28, 28, 1)
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(Conv2D(64, (2, 2), padding='valid'))
model.add(Dropout(0.2))

model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

model.summary()

#3 Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=2,
          validation_split=0.02, callbacks=[es])
duration_time = time.time() - start_time

#4 Evaluate
ic('================= EVALUATE ==================')
loss = model.evaluate(x_test, y_test)   # evaluate -> return loss, metrics
ic(duration_time)
ic('loss', loss[0])
ic('acc',loss[1])

'''
CNN
# loss: 0.05057989060878754
# accuracy: 0.9922999739646912

DNN
ic| duration_time: 13.754892587661743
ic| 'loss', loss[0]: 0.15311706066131592
ic| 'acc', loss[1]: 0.9562000036239624

DNN+CNN
ic| duration_time: 930.8541400432587
ic| 'loss', loss[0]: 0.17542101442813873
ic| 'acc', loss[1]: 0.968999981880188
'''