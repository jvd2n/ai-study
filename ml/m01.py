from icecream import ic
import numpy as np
from sklearn.datasets import load_iris

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

ic(x.shape, y.shape)    # (150, 4) (150,)
ic(y)

# One-Hot-Encoding  (150,) -> (150, 3)
# 0 -> [1, 0, 0]
# 1 -> [0, 1, 0]
# 2 -> [0, 0, 1]

# [0, 1, 2, 1] -> 
# [[1, 0, 0]
#  [0, 1, 0]
#  [0, 0, 1]
#  [0, 1, 0]]   (4,) -> (4, 3)

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# ic(y[:5])
# ic(y.shape)
# ML 기법은 대부분 원핫인코딩 된 것이 아닌 그대로를 받아들임?


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    shuffle=True,
    random_state=66
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC

model = LinearSVC()
# model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(4,)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(3, activation='softmax'))   # softmax -> 다중분류

# 3. 컴파일, 훈련
model.fit(x_train, y_train)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# from tensorflow.keras.callbacks import EarlyStopping
# # es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)
# es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

# hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('model.score : ', results)
# loss = model.evaluate(x_test, y_test)   # evaluate -> return loss, metrics
# print(f'loss: {loss[0]}')
# print(f'accuracy: {loss[1]}')

from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score : ', acc)

print('================= PREDICT =================')
print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)
