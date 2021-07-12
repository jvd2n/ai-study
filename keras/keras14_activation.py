from icecream import ic
import numpy as np
# import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

ic(x.shape, y.shape)  # (442, 10), (442,)

ic(datasets.feature_names)  
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
ic(datasets.DESCR)

ic(y[:30])
ic(np.min(y), np.max(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

#2. 모델 구성
# 모델링은 역삼각 노드, 다이아 노드, 모래시계 노드 등이 일반적이다.
model = Sequential()
model.add(Dense(128, input_dim=10, activation='selu'))
model.add(Dense(256, activation='selu'))
model.add(Dense(512, activation='selu'))
model.add(Dense(256, activation='selu'))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))
# activation
# 1. sigmoid : 0과 1 사이로 한정시킴
# 2. relu : 현재 고성능으로 알려진 활성함수

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=5, validation_split=0.2)

#4. 평가, 예측
# mse, R2 -> R2 값을 1에 가깝게
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)  # x_test를 훈련시킨 값으로
# ic(y_predict)

# R2 결정 계수
r2 = r2_score(y_test, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산
ic(r2)


# 과제1. r2 계수 0.62 이상으로 만들 것