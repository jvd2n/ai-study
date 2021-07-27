from icecream import ic
import numpy as np
from tensorflow.python.keras.saving.save import load_model


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

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# mae(Mean Absolute Error) : 평균 절대 오차

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1,
                   restore_best_weights=True)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True, filepath='./_save/ModelCheckPoint/keras49_MCP.h5')

model.save('./_save/ModelCheckPoint/keras49_model_save.h5')

model.fit([x1_train, x2_train], y_train, epochs=300, batch_size=4, validation_split=0.2, verbose=1, callbacks=[es, mcp])


#4. 평가, 예측
from sklearn.metrics import r2_score
print('=================== 1. original results =======================')
results = model.evaluate([x1_test, x2_test], y_test)

print('loss : ', results[0])
# print("metrics['mae'] : ", results[1])

y_predict = model.predict([x1_test, x2_test])

r2 = r2_score(y_test, y_predict)
print("r2score: ", r2)

print('=================== 2. load_model =======================')
model2 = load_model('./_save/ModelCheckPoint/keras49_model_save.h5')

results = model2.evaluate([x1_test, x2_test], y_test)
print('loss : ', results[0])

y_predict = model2.predict([x1_test, x2_test])

r2 = r2_score(y_test, y_predict)
print("r2score: ", r2)

print('=================== 3. Model Check Point =======================')
model3 = load_model('./_save/ModelCheckPoint/keras49_MCP.h5')

results = model3.evaluate([x1_test, x2_test], y_test)
print('loss : ', results[0])

y_predict = model3.predict([x1_test, x2_test])

r2 = r2_score(y_test, y_predict)
print("r2score: ", r2)

'''
# restore_best_weight=True
loss :  0.00012859962589573115
r2score:  0.9999998528368221
loss :  1131976.375
r2score:  -1293.6012997969478
loss :  0.00012859962589573115
r2score:  0.9999998528368221

# restore_best_weight=False
loss :  3.1292437085994607e-08
r2score:  0.999999999964212
loss :  1103863.125
r2score:  -1261.4491475389611
loss :  3.588696273482128e-08
r2score:  0.9999999999589574
'''