from icecream import ic
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
# import tensorflow as tf

# 1. 데이터
x_train = np.array([1, 2, 3, 4, 5, 6, 7])
y_train = np.array([1, 2, 3, 4, 5, 6, 7])
x_test = np.array([8, 9, 10])
y_test = np.array([8, 9, 10])

# 2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
ic('loss : ', loss)

y_predict = model.predict([11])
ic(y_predict)
