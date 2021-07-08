from icecream import ic
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 4, 3, 5, 7, 9, 3, 8, 12])

# 2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(20))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=1000, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
ic('loss : ', loss)

# result = model.predict([11])
# ic('[11]의 예측값 : ', result)

y_predict = model.predict(x)
ic(y_predict)

plt.scatter(x, y)
plt.plot(x, y_predict, color='red')
plt.show()
