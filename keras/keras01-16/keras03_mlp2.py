from icecream import ic
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 양 끝의 [] 는 제외하고 셈
# 1 : [1, 2, 3]                   >> []를 제거하면 스칼라 3개의 값을 가진 벡터 1개가 된다. (3, )
# 1.1 : [[1],[2],[3]]             >> 3행 1열
# 2 : [[1, 2, 3]]                 >> 1행 3열
# 3 : [[1, 2], [3, 4], [5. 6]]    >> 3행 2열
# 4 : [[[1, 2, 3], [4, 5, 6]]]    >> 1면 2행 3열
# 5 : [[[1, 2], [3, 4], [5, 6]]]  >> 1면 3행 2열
# 6 : [[[1], [2]], [[3], [4]]]    >> 2면 2행 1열

# 행 무시, 열 우선

# 1. 데이터
x = np.array([
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
  [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
  [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
  ]) # 2행 10열
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

# ic(x.shape)
# ic(x1)
x = np.transpose(x) # 행의 개수가 서로 동일해야 하므로
ic(y.shape) # (10,)
ic(x)
ic(x.shape) # (10, 3)

# 2. 모델
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))

# 3
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=2000, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
ic('loss : ', loss)

x_pred = np.array([[11, 1.4, 2]])
# ic(x_pred.shape) # (1, 3)
result = model.predict(x_pred)
ic('x_pred의 예측값 : ', result)