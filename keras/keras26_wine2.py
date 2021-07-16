from icecream import ic 
import numpy as np
import pandas as pd

datasets = pd.read_csv('D:\study\_data\winequality-white.csv', sep=';', index_col=None, header=0)

# ./ : 현재폴더
# ../ : 상위폴더

ic(type(datasets))
ic(datasets.shape)   # (4898, 12)
ic(datasets.info)
ic(datasets.describe())
ic(datasets.index)

#1 pandas -> numpy
#2 x와 y 분리
#3 sklearn의 onehot 사용할 것
#4 y의 라벨을 확인 np.unique(y)
#5 y의 shape 확인 (4898,) -> (4898, 7)

datasets_np = datasets.to_numpy()

ic(datasets_np)
x = datasets_np[:,0:11]
ic(x)
y = datasets_np[:,[-1]]
ic(y)

from sklearn import preprocessing
onehot_enc = preprocessing.OneHotEncoder()
onehot_enc.fit(y)
y = onehot_enc.transform(y).toarray()
ic(y)

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# ic(y)
# ic(y.shape)

ic(x.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.997,
                                                    shuffle=True,
                                                    random_state=15)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
ic(x_train)
ic(x_test)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(11,), sparse=True)
xx = Dense(256, activation='relu')(input1)
xx = Dense(128, activation='relu')(xx)
xx = Dense(128, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
output1 = Dense(7, activation='softmax')(xx)

model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=2, validation_split=0.00003, callbacks=[es])

# print(hist)
# <tensorflow.python.keras.callbacks.History object at 0x000001EA87749CA0>

# print(hist.history.keys())
# print('=============== loss ================')
# print(hist.history['loss'])
# print('=============== val_loss ================')
# print(hist.history['val_loss'])

# 4. 평가, 예측
ic('================= EVALUATE ==================')
loss = model.evaluate(x_test, y_test)   # evaluate -> return loss, metrics
print(f'loss: {loss[0]}')
print(f'accuracy: {loss[1]}')
# loss: 0.11009635776281357
# accuracy: 0.9814814925193787

# ic('================= PREDICT =================')
# ic(y_test[:5])
# y_predict = model.predict(x_test[:5])
# ic(y_predict)


# y_predict = model.predict(x_test)  # x_test를 훈련시킨 값으로
# # ic(y_predict)

# # R2 결정 계수
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산
# ic(r2)

# import matplotlib.pyplot as plt
# # from matplotlib import font_manager, rc

# plt.rc('font', family='NanumGothic')
# # print(plt.rcParams['font.family'])
# plt.plot(hist.history['loss'])   # x: epoch / y: hist.history['loss']
# plt.plot(hist.history['val_loss'])

# plt.title('로스, 발로스')
# plt.xlabel('epochs')
# plt.ylabel('loss, val_loss')
# plt.legend(['train loss', 'val loss'])
# plt.show()

# Standard Scaler
# loss: 2.754335641860962
# accuracy: 0.6340135931968689

# Robust Scaler
# loss: 2.636126756668091
# accuracy: 0.6346938610076904

# loss: 3.1227855682373047
# accuracy: 0.7387754917144775

# loss: 0.7924945950508118
# accuracy: 0.7714285850524902

# loss: 0.6401194930076599
# accuracy: 0.8285714387893677

# loss: 0.4174273908138275
# accuracy: 0.9333333373069763