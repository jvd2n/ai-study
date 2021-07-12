from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from icecream import ic

# x = [1, 2, 3, 4, 5]
# y = [1, 2, 4, 3, 5]
# x_pred = [6]

# 1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 3, 5])

# 2. 모델
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='relu'))

# 3. 컴파일, 훈련
model.compile(loss='msle', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=100)

# 4. 평가, 예측
loss = model.evaluate(x, y)
ic(loss)
y_predict = model.predict(x)
ic('x의 예측값 : ', y_predict)

r2 = r2_score(y, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산
ic(r2)

# R2를 0.9 이상으로 올려라
# loss: 0.022312873974442482
# r2: 0.8020721065487166