import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

#1. Data
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape, y.shape) # (4, 3) (4,)

x = x.reshape(4, 3, 1)  # (batch_size, timesteps, feature)

#2. Modeling
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))
model.add(LSTM(units=10, activation='relu', input_shape=(3, 1)))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)

#4. Evaluate, Predict
x_input = np.array([5,6,7]).reshape(1,3,1)
results = model.predict(x_input)
print(results)  # [[8.000002]] 

model.summary()

'''
(Input + bias) * output + ouput * output
= (Input + bias + output) * output

LSTM -> 4배의 연산이 더 이루어짐

(num_features + num_units)* num_units + biases
param = num_units * * (num_units + input_dim + 1)
파라미터 아웃값 * (파라미터 아웃값 + 디멘션값 + 1(바이어스))
( (unit 개수 + 1) * unit 개수 ) + ( input_dim(feature) 수 * unit 개수 )
'''