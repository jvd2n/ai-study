"""
PATH -> ./_save/_graph/
run -> tensorboard -- logdir=.
"""

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,4,3,5,6,7,8,9,10])


# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1)) # Dense layer y = wx + b
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import TensorBoard
tb = TensorBoard(log_dir='./_save/_graph/', histogram_freq=0, write_graph=True, write_images=True)
model.fit(x, y, epochs=50, batch_size=1, validation_split=0.2, callbacks=[tb])

# 4. 평가, 예측
loss = model.evaluate(x, y)
ic(loss)

y_predict = model.predict([11])
ic(y_predict)
