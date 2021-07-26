from icecream import ic
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 4, 3, 5, 7, 9, 3, 8, 12])

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(20))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)

loss = model.evaluate(x, y)
ic('loss : ', loss)

result = model.predict([11])
ic('[11]의 예측값 : ', result)

# y_predict = model.predict(x)
ic(x, x.shape)
x_list = x.tolist()
y_predict = model.predict(x)
ic(x_list)
ic(y_predict, y_predict.shape)

plt.scatter(x, y)
plt.plot(x, y_predict, color='red')
plt.show()
