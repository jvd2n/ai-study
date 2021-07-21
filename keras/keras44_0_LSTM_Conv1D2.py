# 1~100까지의 데이터

#       x           y
# 1, 2, 3, 4, 5     6
# ...
# 95,96,97,98,99    100

from icecream import ic
import numpy as np
x_data = np.array(range(1, 101))
x_predict = np.array(range(96, 105))

#       x
# 96, 97, 98, 99, 100        ?
# ...
# 101, 102, 103, 104, 105   ?

# 예상 결과값 : 101 102 103 104 105 106

size = 6
# print(len(a))

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(x_data, size)
# print(dataset)

x = dataset[:, :5].reshape(95, 5, 1)
y = dataset[:, 5]

print("x: \n", x)
print("y: ", y)
# print(x.shape)  # (95, 5, 1)

#2 Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

model = Sequential()
model.add(Conv1D(64, kernel_size=2, input_shape=(5, 1)))
model.add(LSTM(64, return_sequences=True))
model.add(Conv1D(64, 2))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))

model.summary()






'''
x_predict = np.array([102, 103, 104, 105])
y_predict = np.array([106])


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_predict = x_predict.reshape(1, x_predict.shape[0], 1)
ic(x_train.shape, x_test.shape)
ic(y_train.shape, y_test.shape)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, SimpleRNN, LSTM, GRU
input1 = Input(shape=(4, 1))
xx = LSTM(32, activation='relu')(input1)
xx = Dense(64, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
xx = Dense(8, activation='relu')(xx)
output1 = Dense(1)(xx)

model = Model(inputs=input1, outputs=output1)

#3. Compile, Train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)

#4. Evaluate, Predict
loss = model.evaluate(x_test, y_test)
ic('loss : ', loss)

y_predict = model.predict(x_test)
ic('x_test의 예측값 : ', y_predict)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
ic(r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.sqrt -> root를 씌움

rmse = RMSE(y_test, y_predict)
ic(rmse)
'''