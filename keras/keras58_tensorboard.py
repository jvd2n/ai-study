from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils

#1. Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ic(x_train.shape, y_train.shape)
# ic(x_test.shape, y_test.shape)
# # ic| x_train.shape: (60000, 28, 28), y_train.shape: (60000,)
# # ic| x_test.shape: (10000, 28, 28), y_test.shape: (10000,)

# ic(x_train[0])
# ic(y_train[0])

# plt.imshow(x_train[111], 'gray')
# plt.show()

# Data preprocessing
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
# (60000, 28, 14, 2) or (60000, 14, 14, 4) 등으로 변경해도 무방
# 단, 모델링의 정교함?이 필요
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

ic(np.unique(y_train))  
# ic| np.unique(y_train): array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)
ic(x_train.shape, x_test.shape)

y_train = y_train.reshape(60000, 1)
y_test = y_test.reshape(10000, 1)
ic(y_train.shape, y_test.shape)

# from sklearn import preprocessing
# onehot_enc = preprocessing.OneHotEncoder()
# onehot_enc.fit(y_train)
# y_train = onehot_enc.transform(y_train).toarray()
# y_test = onehot_enc.transform(y_test).toarray()
# ic(y_train.shape, y_test.shape)


#2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

model = Sequential()
model.add(Conv2D(filters=30, kernel_size=(2, 2), padding='same', input_shape=(28, 28, 1)))
model.add(Dropout(0.8))
model.add(Conv2D(20, (2, 2), activation='relu'))            # (N, 9, 9, 20)
model.add(Conv2D(30, (2, 2), padding='valid'))              # (N, 8, 8, 30)
model.add(MaxPool2D())                                      # (N, 4, 4, 30)
model.add(Conv2D(15, (2,2)))
model.add(Flatten())                                        # (N, 480)
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()

#3 Compile, Train   metrics=['accuracy']
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es = EarlyStopping(monitor='val_loss', patience=7, mode='min', verbose=1)
tb = TensorBoard(log_dir='./_save/_graph/', histogram_freq=0, write_graph=True, write_images=True)

import time
start_time = time.time()
model.fit(x_train, y_train, epochs=20, batch_size=256, verbose=1,
          validation_split=0.25, callbacks=[es, tb])
duration_time = time.time() - start_time


#4 Evaluate
loss = model.evaluate(x_test, y_test)   # evaluate -> return loss, metrics
ic(duration_time)
print(f'loss: {loss[0]}')
print(f'accuracy: {loss[1]}')


'''
loss: 0.05057989060878754
accuracy: 0.9922999739646912

loss: 0.09428535401821136
accuracy: 0.9919000267982483
'''
