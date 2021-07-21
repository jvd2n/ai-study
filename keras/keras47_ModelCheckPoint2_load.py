import time
import numpy as np
from icecream import ic

#1. 데이터
from sklearn.datasets import load_diabetes
datasets = load_diabetes()
x = datasets.data
y = datasets.target

ic(x.shape, y.shape)  # (442, 10), (442,)

ic(datasets.feature_names)  
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
ic(datasets.DESCR)

ic(y[:30])
ic(np.min(y), np.max(y))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=10)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
# model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(10,))) 
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1))

# model.summary()

# 3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', 
#                      filepath='./_save/ModelCheckPoint/keras47_MCP.hdf5')

# start_time = time.time()
# model.fit(x_train, y_train, epochs=30, validation_split=0.2, callbacks=[es, cp])
# duration_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras47_model_save.h5')

# model = load_model('./_save/ModelCheckPoint/keras47_model_save.h5')   # save_model
model = load_model('./_save/ModelCheckPoint/keras47_MCP.hdf5')  # CheckPoint

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
# ic(y_predict)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)    # y_test와 y_predict값을 통해 결정계수를 계산
# ic(duration_time)
ic(loss)
ic(r2)

'''
# ic| loss: 2090.3447265625
# ic| r2: 0.6538537876822268

save_model Result
ic| duration_time: 2.997481107711792
ic| loss: 2990.705322265625
ic| r2: 0.5215782249145064

load_model Result
ic| duration_time: 2.97646164894104
ic| loss: 3000.410888671875
ic| r2: 0.5200257452751009

MCP Result
ic| loss: [3034.45458984375, 43.94503402709961, 0.0]
ic| r2: 0.5145797044170701

load_model.h5
ic| loss: [3034.45458984375, 43.94503402709961, 0.0]
ic| r2: 0.5145797044170701

MCP.hdf5
ic| loss: [3045.30908203125, 44.01026916503906, 0.0]
ic| r2: 0.5128433409813429
'''