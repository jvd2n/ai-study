import numpy as np

#1 Data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,4,7,6,2,8,9,10])

#2 Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3 Compile, Train
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta, RMSprop, SGD, Nadam

# optimizer = Adam(learning_rate=0.01)
# optimizer = Adam(learning_rate=0.001)
# optimizer = Adam(learning_rate=0.0001)
# loss:  3.0650699138641357 result:  [[10.020428]]
# loss:  3.5026919841766357 result:  [[11.038669]]
# loss:  3.086477518081665 result:  [[10.158133]]
# optimizer = Adagrad(learning_rate=0.01)
# optimizer = Adagrad(learning_rate=0.001)
# optimizer = Adagrad(learning_rate=0.0001)
# loss:  3.247929334640503 result:  [[9.216648]]
# loss:  3.126508951187134 result:  [[10.262238]]
# loss:  3.2420754432678223 result:  [[10.419776]]
# optimizer = Adamax(learning_rate=0.01)
# optimizer = Adamax(learning_rate=0.001)
# optimizer = Adamax(learning_rate=0.0001)
# loss:  3.28309965133667 result:  [[10.670873]]
# loss:  4.1093854904174805 result:  [[8.176195]]
# loss:  3.179327964782715 result:  [[10.589341]]
# optimizer = Adadelta(learning_rate=0.01)
# optimizer = Adadelta(learning_rate=0.001)
# optimizer = Adadelta(learning_rate=0.0001)
# loss:  3.235276460647583 result:  [[10.549751]]
# loss:  14.270563125610352 result:  [[4.575527]]
# loss:  29.123409271240234 result:  [[1.5024698]]
# optimizer = RMSprop(learning_rate=0.01)
# optimizer = RMSprop(learning_rate=0.001)
# optimizer = RMSprop(learning_rate=0.0001)
# loss:  21.845447540283203 result:  [[1.8594289]]
# loss:  3.0497307777404785 result:  [[9.85771]]
# loss:  3.1314587593078613 result:  [[10.456734]]
optimizer = SGD(learning_rate=0.01)
# optimizer = SGD(learning_rate=0.001)
# optimizer = SGD(learning_rate=0.0001)
# loss:  nan result:  [[nan]]
# loss:  3.111556053161621 result:  [[10.211786]]
# loss:  3.13327956199646 result:  [[10.365079]]
# optimizer = Nadam(learning_rate=0.01)
# optimizer = Nadam(learning_rate=0.001)
# optimizer = Nadam(learning_rate=0.0001)
# loss:  4.129075527191162 result:  [[11.771722]]
# loss:  3.3699231147766113 result:  [[8.883714]]
# loss:  3.105130195617676 result:  [[10.338067]]

# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)

#4 evaluate, predict
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])

print('loss: ', loss, 'result: ', y_pred)
'''
loss:  5.230731010437012 result:  [[12.450352]]
loss:  3.051877737045288 result:  [[9.942625]]
'''