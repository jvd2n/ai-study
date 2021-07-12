from icecream import ic
import numpy as np
# import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

ic(x.shape, y.shape)  # (442, 10), (442,)

ic(datasets.feature_names)  
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
ic(datasets.DESCR)

ic(y[:30])
ic(np.min(y), np.max(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=10)

#2. 모델 구성
# 모델링은 역삼각 노드, 다이아 노드, 모래시계 노드 등이 일반적이다.
# optimizers=['adam']
# activations=['elu', 'relu', 'linear']

# r2score = []
# opt = []
# act = []
# y_pred = []
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# early_stopping = EarlyStopping(monitor='val_loss', patience=60, mode='min')

# model = Sequential()
# for i in range(len(optimizers)):
#     for j in range(len(activations)):
#         model.add(Dense(256,activation=activations[j], input_shape=(10,)))
#         model.add(Dense(128,activation=activations[j]))
#         model.add(Dense(64,activation=activations[j]))
#         model.add(Dense(32,activation=activations[j]))
#         model.add(Dense(16,activation=activations[j]))
#         model.add(Dense(1))
#         model.compile(loss = 'mae', optimizer=optimizers[i])
#         model.fit(x_train, y_train, epochs=500,
#                   batch_size=32, validation_split=0.2, callbacks=[early_stopping])
        
#         loss = model.evaluate(x_test, y_test)
#         y_predict = model.predict(x_test)
#         y_pred.append(y_predict)
        
#         r2 = r2_score(y_test, y_predict)
#         r2score.append(r2)
        
#         opt.append(optimizers[i])
#         act.append(activations[j])

# print("loss : ", loss)
# y_predict = model.predict(x_test)
# index = r2score.index(max(r2score))
# print("x value : ", y_pred[index])
# print("best optimizer : ", opt[index],", best activation : ",act[index],", r2score : ", r2score[index])

model = Sequential()
model.add(Dense(256, input_shape=(10,),activation='selu')) 
model.add(Dense(230,activation='selu'))
model.add(Dense(190,activation='selu'))
model.add(Dense(140,activation='selu'))
model.add(Dense(130,activation='selu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.01, shuffle=True)

# 4. 평가, 예측
# mse, R2 -> R2 값을 1에 가깝게
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)  # x_test를 훈련시킨 값으로
# ic(y_predict)

# R2 결정 계수
r2 = r2_score(y_test, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산
ic(r2)


# 과제1. r2 계수 0.62 이상으로 만들 것
# ic| loss: 2121.56982421875
# ic| r2: 0.6180938756228092