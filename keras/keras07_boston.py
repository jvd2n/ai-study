from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from icecream import ic
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape)
print(y.shape)

print(datasets.feature_names)
# print(datasets.DESCR)

ic(x)
ic(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=66)
ic(x_test, y_test)

model = Sequential()
model.add(Dense(506, input_dim=13))
model.add(Dense(340, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='relu'))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
ic('loss : ', loss)

y_predict = model.predict(x_test)  # x_test를 훈련시킨 값으로
ic('x_test의 예측값 : ', y_predict)


# R2 결정 계수

r2 = r2_score(y_test, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산
ic('R2 스코어 : ', r2)
# 완료하시오!
