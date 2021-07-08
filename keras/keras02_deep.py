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
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=3000, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
ic('loss : ', loss)

result = model.predict([6])
ic('x_pred [6]의 예측값 : ', result)
