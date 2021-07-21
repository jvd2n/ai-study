# 1~100까지의 데이터

#       x           y
# 1, 2, 3, 4, 5     6
# ...
# 95,96,97,98,99    100

from icecream import ic
import numpy as np
x_data = np.array(range(1, 101))
x_predict = np.array(range(96, 106))

#       x
# 96, 97, 98, 99, 100        ?
# ...
# 101, 102, 103, 104, 105   ?

# 예상 결과값 : 101 102 103 104 105 106

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(x_data, 6)
x_predict = split_x(x_predict, 5)
# print(dataset)

x = dataset[:, :5]
y = dataset[:, 5]

# print("x: \n", x)
# print("y: ", y)
ic(x_predict)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=66)

ic(x_test.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_predict = scaler.transform(x_predict)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, SimpleRNN, LSTM, GRU
input1 = Input(shape=(5, 1))
xx = LSTM(32, activation='relu')(input1)
xx = Dense(64, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
xx = Dense(8, activation='relu')(xx)
output1 = Dense(1)(xx)

model = Model(inputs=input1, outputs=output1)

#3. Compile, Train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_split=0.02, verbose=2)

#4. Evaluate, Predict
loss = model.evaluate(x_test, y_test)
ic(loss)

# y_predict = model.predict(x_test)
y_predict = model.predict(x_test)
# print('x_test의 예측값 : ', y_predict)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
ic(r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.sqrt -> root를 씌움

rmse = RMSE(y_test, y_predict)
ic(rmse)

'''
ic| loss: 0.2987632155418396
ic| r2: 0.9995578592868813
ic| rmse: 0.5465933636657719
'''