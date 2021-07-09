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
model.add(Dense(10, input_dim=1))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=200, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
ic('loss : ', loss)
y_predict = model.predict(x)
ic('x의 예측값 : ', y_predict)


from sklearn.metrics import r2_score

r2 = r2_score(y, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산
ic('R2 스코어 : ', r2)


#과제2
#R2를 0.9 이상으로 올려라