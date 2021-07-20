from icecream import ic
import numpy as np


#1. 데이터
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)
# y1 = np.array([range(1001, 1101)])
# y1 = np.transpose(y1)   # (100, 1)
y = np.array(range(1001, 1101))
ic(x1.shape, x2.shape, y.shape) # (100, 3) (100, 3) (100,)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.7, random_state=66)

ic(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, y_train.shape, y_test.shape)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu', name='dense1')(input1)
dense2 = Dense(3, activation='relu', name='dense2')(dense1)
dense3 = Dense(2, activation='relu', name='dense3')(dense2)
output1 = Dense(3, name='output1')(dense3)

#2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(4, activation='relu', name='dense11')(input2)
dense12 = Dense(4, activation='relu', name='dense12')(dense11)
dense13 = Dense(4, activation='relu', name='dense13')(dense12)
dense14 = Dense(4, activation='relu', name='dense14')(dense13)
output2 = Dense(4, name='output2')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate
# merge1 = concatenate([output1, output2])
merge1 = Concatenate(axis=1)([output1, output2])
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)
last_output = Dense(1)(merge3)

# model = Model(inputs=[input1, input2], outputs=[output1, output2])
model = Model(inputs=[input1, input2], outputs=last_output)


model.summary()
# 번갈아가며 연산이 되는 것처럼 보이지만 concatenate 전에는 모델 간에 영향을 미치지 않음


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# mae(Mean Absolute Error) : 평균 절대 오차
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=8, verbose=1)


#4. 평가, 예측
results = model.evaluate([x1_test, x2_test], y_test)
# ic(results)
print('loss : ', results[0])
print("metrics['mae'] : ", results[1])


#1 도박사의 오류 3-4줄 정리
#2 train_size의 디폴트 찾기
#3 평균값과 중위값의 차이
#4 Concatenate로 바꿔 코딩 하시오
# merge1 = Concatenate(axis=1)([output1, output2])