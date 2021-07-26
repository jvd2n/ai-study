from icecream import ic
from sklearn.datasets import load_boston
import numpy as np

datasets = load_boston()
x = datasets.data
y = datasets.target
# ic(datasets.feature_names)
# ic(datasets.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=66)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

ic(x_train.shape, x_test.shape)

x_train = x_train.reshape(-1, x_train.shape[1], 1)
x_test = x_test.reshape(-1, x_test.shape[1], 1)

ic(x_train.shape, x_test.shape)
ic(y_train.shape, y_test.shape)

# 2. model 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Dropout, GlobalAveragePooling1D, MaxPooling1D

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu', input_shape=(13, 1)))
model.add(Dropout(0.2))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es])
duration_time = time.time() - start_time

# 4. 평가 예측
from sklearn.metrics import r2_score

y_predict = model.predict([x_test])
# print('x의 예측값 : ', y_predict)
loss = model.evaluate(x_test, y_test)
r2 = r2_score(y_test, y_predict)

ic(duration_time)
ic(loss)
ic(r2)

'''
Robust DNN
ic| loss: 7.366783142089844
ic| r2: 0.9118625981667393

Robust CNN
DoT: 17.160114288330078
loss: 6.867620944976807
r2: 0.9178346577419654

Robust LSTM
ic| duration_time: 9.910001516342163
ic| loss: 14.023414611816406
ic| r2: 0.8313014470641965

Robust Conv1D
ic| duration_time: 13.306169509887695
ic| loss: 6.393956661224365
ic| r2: 0.9235016504830248
'''