import numpy as np
from icecream import ic

# 1. 데이터
x = np.array([range(100), range(301,401), range(1, 101),
              range(100), range(401,501)])
x = np.transpose(x)

ic(x.shape)
y = np.array([range(711, 811), range(101, 201)])
y = np.transpose(y)
ic(y.shape)

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 함수형 모델은 좀 더 유연하게 변경이 가능하다. 시작점과 끝지점만 잡아주면 된다.
input1 = Input(shape=(5,))
dense1 = Dense(3)(input1)
dense2 = Dense(4)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(2)(dense3)

model = Model(inputs=input1, outputs=output1)

# Sequential 모델은 이후에 여러 모델을 엮을 필요가 있는 경우 한계가 있음
# 단일 모델을 사용하는 경우 편리함
# model = Sequential()
# model.add(Dense(3, input_shape=(5,)))
# model.add(Dense(4))
# model.add(Dense(10))
# model.add(Dense(2))

model.summary()

# 3. 컴파일, 훈련

# 4. 평가 예측
