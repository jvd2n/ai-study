from icecream import ic
from sklearn.datasets import load_boston
import numpy as np

datasets = load_boston()
x = datasets.data
y = datasets.target


ic(datasets.feature_names)
# ic(datasets.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=66)
# ic(x_test, y_test)
# ic(x.shape)         # (506, 13)
# ic(x_train.shape)   # (354, 13)
# ic(x_test.shape)    # (152, 13)
# ic(x_train)
# ic(y_train)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[es])

# print(hist)
# <tensorflow.python.keras.callbacks.History object at 0x000001EA87749CA0>

print(hist.history.keys())
print('=============== loss ================')
print(hist.history['loss'])
print('=============== val_loss ================')
print(hist.history['val_loss'])

# 4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# ic(loss)

# y_predict = model.predict(x_test)  # x_test를 훈련시킨 값으로
# # ic(y_predict)

# # R2 결정 계수
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산
# ic(r2)

import matplotlib.pyplot as plt

plt.rc('font', family='NanumGothic')
print(plt.rcParams['font.family'])
plt.plot(hist.history['loss'])   # x: epoch / y: hist.history['loss']
plt.plot(hist.history['val_loss'])

plt.title('로스, 발로스')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend(['train loss', 'val loss'])
plt.show()

# minmax 전처리 후
# ic| loss: 13.08879566192627
# ic| r2: 0.8415728130024938

# MinMaxScaler 전처리 후
# ic| loss: 6.698724746704102
# ic| r2: 0.9189184225778089
# ic| loss: 5.6741437911987305
# ic| r2: 0.9313199876905316

# MinMaxScaler & Train/Test Scale
# ic| loss: 5.566455364227295
# ic| r2: 0.9326234522046936

# train_size=0.8 & StandardScaler
# ic| loss: 7.42013692855835
# ic| r2: 0.9112242701815445