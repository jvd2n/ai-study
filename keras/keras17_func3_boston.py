from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from icecream import ic
from sklearn.datasets import load_boston

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

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# model = Sequential()
# model.add(Dense(10, input_dim=13))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(3, activation='relu'))
# model.add(Dense(1, activation='relu'))
# loss: 15.07454776763916
# r2: 0.8175371850113262

input1 = Input(shape=(13,))
dense1 = Dense(32)(input1)
dense2 = Dense(32, activation='relu')(dense1)
dense3 = Dense(16, activation='relu')(dense2)
dense4 = Dense(8, activation='relu')(dense3)
dense5 = Dense(4, activation='relu')(dense4)
dense6 = Dense(2, activation='relu')(dense5)
output1 = Dense(1, activation='relu')(dense6)

model = Model(inputs=input1, outputs=output1)
# ic| loss: 14.213667869567871
# ic| r2: 0.8279573196329961

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)  # x_test를 훈련시킨 값으로
ic(y_predict)

# R2 결정 계수
r2 = r2_score(y_test, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산
ic(r2)

model.summary()