from icecream import ic
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 1 데이터
x = np.array([range(10)])
# ic(x.shape)
x = np.transpose(x)

y = np.array([
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
  [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
  [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
  ])  # (3, 10)
y = np.transpose(y)
# ic(y.shape) # (10, 3)

# 2. 모델
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(14))
model.add(Dense(6))
model.add(Dense(3))

# 3
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=1000, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
ic('loss : ', loss)

x_pred = np.array([[9]])
# ic(x_pred.shape) # (1, 3)
result = model.predict(x_pred)
ic('x_pred의 예측값 : ', result)