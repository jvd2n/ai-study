from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

# train_test_split으로 만들기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, shuffle=False, random_state=66)
# ic(x_train, x_test, x_val, y_train, y_test, y_val)

# x_train = np.array([1, 2, 3, 4, 5, 6, 7]) # 훈련, 학습 시키는 것
# y_train = np.array([1, 2, 3, 4, 5, 6, 7])
# x_test = np.array([8, 9, 10]) # 실질적으로 평가하는 것
# y_test = np.array([8, 9, 10])
# x_val = np.array([11, 12, 13])
# y_val = np.array([11, 12, 13])

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1)) # Dense layer y = wx + b
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.3)
# loss는 통상적으로 val_loss보다 과적합에 잘 걸림

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict([11])
ic(y_predict)
