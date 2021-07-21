from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from icecream import ic

datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape)
print(y.shape)

print(datasets.feature_names)
# print(datasets.DESCR)

ic(x)
ic(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)
ic(x_test, y_test)

model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(20, activation='selu'))
model.add(Dense(10, activation='selu'))
model.add(Dense(10, activation='selu'))
model.add(Dense(5, activation='selu'))
model.add(Dense(3, activation='selu'))
model.add(Dense(1, activation='selu'))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.3)
# loss는 test에서 바로 나오는 값
# val_loss는 문제를 겪어가면서 추출하는 값으로, epoch이 높아져도 낮아지는 폭이 적으므로 과적합이 될 확률이 낮음

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)  # x_test를 훈련시킨 값으로
ic(y_predict)

# R2 결정 계수
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산
ic(r2)

# loss: 15.07454776763916
# r2: 0.8175371850113262
