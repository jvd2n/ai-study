from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from keras.utils import np_utils

# 1. Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, PowerTransformer
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(32 * 32 * 3,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3 Compile, Train   metrics=['accuracy']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=2, validation_split=0.025, callbacks=[es])


# 4 Evaluate
ic('================= EVALUATE ==================')
loss = model.evaluate(x_test, y_test)   # evaluate -> return loss, metrics
print(f'loss: {loss[0]}')
print(f'accuracy: {loss[1]}')

'''
CNN
loss: 4.017408847808838
accuracy: 0.6402000188827515

DNN
loss: 1.5292080640792847
accuracy: 0.4584999978542328
'''
