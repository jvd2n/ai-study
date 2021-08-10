import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 data
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

#2 model
# model = SVC()
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3 compile, train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4 evaluate, predict
y_pred = model.predict(x_data)
print(x_data, '의 예측 결과 : ', y_pred)

results = model.evaluate(x_data, y_data)
print('model.score : ', results[1]) # [0]: loss, [1]: acc

acc = accuracy_score(y_data, np.around(y_pred))
print('acc_score : ', acc)