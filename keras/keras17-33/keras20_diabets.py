from icecream import ic
import numpy as np
from sklearn.datasets import load_diabetes

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

ic(x.shape, y.shape)  # (442, 10), (442,)

ic(datasets.feature_names)  
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# ic(datasets.DESCR)

# ic(y[:30])
# ic(np.min(y), np.max(y))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델 구성
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# model = Sequential()
# model.add(Dense(256, activation='selu', input_shape=(10,))) 
# model.add(Dense(256, activation='selu'))
# model.add(Dense(128, activation='selu'))
# model.add(Dense(64, activation='selu'))
# model.add(Dense(32, activation='selu'))
# model.add(Dense(1))
# ic| loss: 2090.3447265625
# ic| r2: 0.6538537876822268

input1 = Input(shape=(10,))
xx = Dense(128, activation='selu')(input1)
xx = Dense(64, activation='selu')(xx)
xx = Dense(64, activation='selu')(xx)
xx = Dense(64, activation='selu')(xx)
xx = Dense(32, activation='selu')(xx)
xx = Dense(32, activation='selu')(xx)
output1 = Dense(1, activation='selu')(xx)

model = Model(inputs=input1, outputs=output1)
# ic| loss: 2057.89306640625
# ic| r2: 0.6592275448715366

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=8)

# 4. 평가, 예측
# mse, R2 -> R2 값을 1에 가깝게
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)  # x_test를 훈련시킨 값으로
# ic(y_predict)

# R2 결정 계수
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산
ic(r2)

# model.summary()

# MinMaxScaler
# ic| loss: 3763.9970703125
# ic| r2: 0.42003486668230217

# StandardScaler
# ic| loss: 4926.21337890625
# ic| r2: 0.24095796424921867
