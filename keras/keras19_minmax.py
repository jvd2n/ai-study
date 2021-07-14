from icecream import ic
from sklearn.datasets import load_boston
import numpy as np

datasets = load_boston()
x = datasets.data
y = datasets.target


ic(datasets.feature_names)
# ic(datasets.DESCR)

# ic(x)
# ic(y)

# 데이터 정규화 (normalization / regulization / minmax scaler)
# 모든 값을 최대값으로 나누어 0~1 사이의 값으로 만든다
# but, 최소값이 0이 아닌 경우 -> 개체-최소값 / 최대값-최소값
ic(np.min(x), np.max(x))    # (0.0, 711.0)
x = x/711.
# ic(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=66)
ic(x_test, y_test)
ic(x.shape)         # (506, 13)
ic(x_train.shape)   # (354, 13)
ic(x_test.shape)    # (152, 13)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=8)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)  # x_test를 훈련시킨 값으로
# ic(y_predict)

# R2 결정 계수
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산
ic(r2)

# ic| loss: 13.08879566192627
# ic| r2: 0.8415728130024938