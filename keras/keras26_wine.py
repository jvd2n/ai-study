from icecream import ic
import numpy as np
from sklearn.datasets import load_wine

datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

ic(x.shape, y.shape)    # (150, 4) (150,)
ic(y)
ic(datasets)
ic(datasets.target_names)


from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
ic(y[:5])
ic(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=66)

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
model.add(Dense(3, activation='softmax'))   # softmax -> 다중분류

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es])

# print(hist)
# <tensorflow.python.keras.callbacks.History object at 0x000001EA87749CA0>

# print(hist.history.keys())
# print('=============== loss ================')
# print(hist.history['loss'])
# print('=============== val_loss ================')
# print(hist.history['val_loss'])

# 4. 평가, 예측
ic('================= EVALUATE =================')
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