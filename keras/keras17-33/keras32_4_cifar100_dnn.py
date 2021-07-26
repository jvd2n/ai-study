from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from keras.utils import np_utils

# 1. Data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)
# x_train = x_train.reshape(50000, 32, 32, 3)
# x_test = x_test.reshape(10000, 32, 32, 3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, PowerTransformer
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32 * 32, 3)
x_test = x_test.reshape(10000, 32 * 32, 3)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D
model = Sequential()
# model.add(Dense(512, activation='relu', input_shape=(32 * 32 * 3,)))
model.add(Dense(512, activation='relu', input_shape=(32 * 32, 3)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(100, activation='softmax'))

# 3 Compile, Train   metrics=['accuracy']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1, validation_split=0.0002, callbacks=[es])


# 4 Evaluate
ic('================= EVALUATE ==================')
loss = model.evaluate(x_test, y_test)   # evaluate -> return loss, metrics
print(f'loss: {loss[0]}')
print(f'accuracy: {loss[1]}')

'''
CNN
# loss: 7.4432373046875
# accuracy: 0.3100999891757965

DNN
loss: 4.313990116119385
accuracy: 0.22339999675750732

DNN + GAP
loss: 3.754659414291382
accuracy: 0.14589999616146088
'''
