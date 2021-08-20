# trainable 동결, 미동결 비교
# fc 모델, average pooling 비교


from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19
from sklearn import preprocessing
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from keras.utils import np_utils

#1 Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

ic(np.unique(y_train))
# ic| np.unique(y_train): array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)
ic(x_train.shape, x_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
# ic(y_train.shape, y_test.shape)

#2 Model
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
vgg16.trainable = False   # vgg16 훈련을 동결한다.

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()

# 3 Compile, Train   metrics=['accuracy']
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=30, batch_size=1024,
          verbose=1, validation_split=0.25, callbacks=[es])

#4 Evaluate
ic('================= EVALUATE ==================')
loss = model.evaluate(x_test, y_test)   # evaluate -> return loss, metrics
print(f'loss: {loss[0]}')
print(f'accuracy: {loss[1]}')

# loss: 1.0598273277282715
# accuracy: 0.642799973487854