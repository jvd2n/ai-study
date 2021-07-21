import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

#1. Data
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])

print(x.shape, y.shape) # (4, 3) (4,)

x = x.reshape(-1, 3, 1)  # (batch_size, timesteps, feature)
x = x.reshape(x.shape[0], x.shape[1], 1)  # (batch_size, timesteps, feature)
x_predict = x_predict.reshape(1, x_predict.shape[0], 1)
print(x.shape, x_predict.shape)
#2. Modeling
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))
model.add(LSTM(units=10, activation='relu', input_shape=(3, 1), return_sequences=True))
model.add(LSTM(units=7, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.summary()

#3. Compile, Train
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, verbose=1)

#4. Evaluate, Predict
results = model.predict(x_predict)
print(results)

'''
[[82.981384]]

(Input + bias) * output + ouput * output
= (Input + bias + output) * output

LSTM -> 4배의 연산이 더 이루어짐

(num_features + num_units)* num_units + biases
param = num_units * * (num_units + input_dim + 1)
파라미터 아웃값 * (파라미터 아웃값 + 디멘션값 + 1(바이어스))
( (unit 개수 + 1) * unit 개수 ) + ( input_dim(feature) 수 * unit 개수 )
'''