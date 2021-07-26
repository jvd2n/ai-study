from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from keras.utils import np_utils

#1. Data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)  # (50000, 3072)
x_test = x_test.reshape(10000, 32 * 32 * 3)    # (10000, 3072)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, PowerTransformer
# scaler = MinMaxScaler()
# scaler = PowerTransformer()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)  # (50000, 3072)
x_test = x_test.reshape(10000, 32, 32, 3)    # (10000, 3072)

# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
# ic(y_train.shape, y_test.shape)

# from sklearn import preprocessing
# onehot_enc = preprocessing.OneHotEncoder()
# onehot_enc.fit(y_train)
# y_train = onehot_enc.transform(y_train).toarray()
# y_test = onehot_enc.transform(y_test).toarray()
# ic(y_train.shape, y_test.shape)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2, 2), 
                padding='valid', activation='relu', 
                input_shape=(32, 32, 3))) 
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))                   
model.add(MaxPool2D())

model.add(Conv2D(128, (2, 2), padding='valid', activation='relu'))                   
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))    
model.add(MaxPool2D())

model.add(Conv2D(64, (2, 2), activation='relu'))                   
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

# model.summary()

#3 Compile, Train   metrics=['accuracy']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)
es = EarlyStopping(monitor='val_loss', patience=7, mode='min', verbose=1)

import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1,
                validation_split=0.25, callbacks=[es])
end_time = time.time() - start_time

#4 Evaluate
ic('================= EVALUATE ==================')
loss = model.evaluate(x_test, y_test)   # evaluate -> return loss, metrics
print(f'DoT: {end_time}')
print(f'loss: {loss[0]}')
print(f'acc: {loss[1]}')
'''
loss: 7.4432373046875
accuracy: 0.3100999891757965

Duration of Time: 91.5438985824585
loss: 5.169971466064453
acc: 0.42010000348091125

batch -> 256, valid_split -> 0.2
DoT: 58.795228719711304
loss: 3.797473907470703
acc: 0.35690000653266907

batch -> 64
DoT: 113.02568006515503
loss: 3.51678466796875
acc: 0.3416000008583069

val 0.2 -> 0.25 , re-modeling
DoT: 151.42099380493164
loss: 2.951824903488159
acc: 0.3700999915599823
'''

import matplotlib.pyplot as plt
plt.figure(figsize=(9, 5))

# 1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()