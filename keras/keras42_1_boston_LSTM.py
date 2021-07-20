import time
from icecream import ic
from sklearn.datasets import load_boston
import numpy as np

datasets = load_boston()
x = datasets.data
y = datasets.target

# ic(datasets.feature_names)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.75,
                                                    shuffle=True,
                                                    random_state=66)
ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)  # (batch_size, timesteps, feature)
x_test = x_test.reshape(-1, x_test.shape[1], 1)
ic(x_train.shape, x_test.shape)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(13, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

start_time = time.time()
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=2, callbacks=[es])
end_time = time.time()
duration_time = end_time - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)  # x_test를 훈련시킨 값으로
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산

ic(duration_time)
ic(loss)
ic(r2)

'''
# MinMaxScaler & Train/Test Scale
# ic| loss: 5.566455364227295
# ic| r2: 0.9326234522046936

LSTM
ic| duration_time: 9.910001516342163
ic| loss: 14.023414611816406
ic| r2: 0.8313014470641965
'''