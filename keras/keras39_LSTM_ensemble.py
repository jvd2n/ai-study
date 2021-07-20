import time
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, SimpleRNN, LSTM, GRU

#1. Data
x1 = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])
x2 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60],
              [50,60,70], [60,70,80], [70,80,90], [80,90,100],
              [90,100,110], [100,110,120],
              [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x1.shape, x2.shape, y.shape)

x1_predict = np.array([55,65,75])
x2_predict = np.array([65,75,85])

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.7, random_state=66)

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)  # (batch_size, timesteps, feature)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)  # (batch_size, timesteps, feature)
x1_predict = x1_predict.reshape(1, x1_predict.shape[0], 1)
x2_predict = x2_predict.reshape(1, x2_predict.shape[0], 1)

#2. Modeling
input1 = Input(shape=(3, 1))
xx = LSTM(64, activation='relu')(input1)
xx = Dense(64, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
xx = Dense(8, activation='relu')(xx)
output1 = Dense(1)(xx)
model = Model(inputs=input1, outputs=output1)

input2 = Input(shape=(3, 1))
xx = LSTM(64, activation='relu')(input2)
xx = Dense(64, activation='relu')(xx)
xx = Dense(64, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
xx = Dense(32, activation='relu')(xx)
xx = Dense(8, activation='relu')(xx)
output2 = Dense(1)(xx)
model = Model(inputs=input2, outputs=output2)

from tensorflow.keras.layers import concatenate, Concatenate
# merge1 = concatenate([output1, output2])
merge1 = Concatenate(axis=1)([output1, output2])
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)
last_output = Dense(1)(merge3)

# model = Model(inputs=[input1, input2], outputs=[output1, output2])
model = Model(inputs=[input1, input2], outputs=last_output)

#3. Compile, Train
start_time = time.time()
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=4, verbose=1)
end_time = time.time() - start_time

#4. Evaluate, Predict
e_results = model.evaluate([x1_test, x2_test], y_test)
p_results = model.predict([x1_predict, x2_predict])
print(f'DoT: {end_time}')
print('loss : ', e_results[0])
print("metrics['mae'] : ", e_results[1])
print("predict: ", p_results)