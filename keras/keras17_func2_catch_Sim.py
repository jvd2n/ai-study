from sklearn.metrics import r2_score
import numpy as np
from icecream import ic

# x = [1, 2, 3, 4, 5]
# y = [1, 2, 4, 3, 5]
# x_pred = [6]

# 1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 3, 5])

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# model = Sequential()
# model.add(Dense(1, input_dim=1))
# model.add(Dense(20, activation='selu'))
# model.add(Dense(30, activation='selu'))
# model.add(Dense(1))

input1 = Input(shape=(1,))
dense1 = Dense(16, activation='selu')(input1)
dense2 = Dense(8, activation='selu')(dense1)
dense3 = Dense(8, activation='selu')(dense2)
dense4 = Dense(4, activation='selu')(dense3)
output1 = Dense(1)(dense4)

model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
ic(loss)
y_predict = model.predict(x)
ic('x의 예측값 : ', y_predict)

r2 = r2_score(y, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산
ic(r2)
model.summary()

# R2를 0.9 이상으로 올려라
# loss: 3.841508259938564e-06
# r2: 0.9999980792456142