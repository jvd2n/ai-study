from icecream import ic 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#1 Data
stock_sk = pd.read_excel('./_data/SK주가 20210721.xls', 
                         index_col=None, header=0, usecols=[0,1,2,3,4,10])
stock_ss = pd.read_excel('./_data/삼성전자 주가 20210721.xls', 
                         index_col=None, header=0, usecols=[0,1,2,3,4,10])


# Preprocessing
stock_sk.columns = ['date','Open','High','Low','Close','Volume']
stock_ss.columns = ['date','Open','High','Low','Close','Volume']
stock_sk = stock_sk.iloc[:2601,:]
stock_ss = stock_ss.iloc[:2601,:]

ic(stock_sk.shape, stock_ss.shape)   # (2601, 5) , (2601, 5)
# stock_sk['date'] = stock_sk['date'].str.replace("/", "")
stock_sk['date'] = pd.to_datetime(stock_sk['date'])
stock_ss['date'] = pd.to_datetime(stock_sk['date'])
stock_sk.set_index('date', inplace=True)
stock_ss.set_index('date', inplace=True)
# ic(stock_sk['date'])
# ic(stock_ss['date'])
# stock_sk = stock_sk.sort_index(ascending=True)
# stock_ss = stock_ss.sort_index(ascending=True)
ic(stock_sk.info)
ic(stock_ss.info)
ic(stock_sk, stock_ss)

from sklearn.preprocessing import MinMaxScaler, RobustScaler
scaler = MinMaxScaler()
scale_cols = ['Open', 'High', 'Low', 'Volume']
scaled_sk = scaler.fit_transform(stock_sk[scale_cols])
scaled_ss = scaler.fit_transform(stock_ss[scale_cols])
ic(scaled_sk)
ic(scaled_ss)

df_sk = pd.DataFrame(scaled_sk, columns=scale_cols)
df_ss = pd.DataFrame(scaled_ss, columns=scale_cols)

from sklearn.model_selection import train_test_split
x_train_sk, x_test_sk, y_train_sk, y_test_sk, x_train_ss, x_test_ss, y_train_ss, y_test_ss = train_test_split(df_sk, stock_sk['Close'], df_ss, stock_ss['Close'], test_size=0.2, random_state=32, shuffle=True)

x_train_sk = x_train_sk.to_numpy()
x_test_sk = x_test_sk.to_numpy()
x_train_ss = x_train_ss.to_numpy()
x_test_ss = x_test_ss.to_numpy()
y_train_sk = y_train_sk.to_numpy()
y_test_sk = y_test_sk.to_numpy()
y_train_ss = y_train_ss.to_numpy()
y_test_ss = y_test_ss.to_numpy()

ic(x_train_sk.shape, y_train_sk.shape)
ic(x_train_ss.shape, y_train_ss.shape)
ic(x_test_sk.shape, y_test_sk.shape)
ic(x_test_ss.shape, y_test_ss.shape)

# ic(x_train_sk, y_train_sk)
# ic(x_train_ss, y_train_ss)
# ic(x_test_sk, y_test_sk)
# ic(x_test_ss, y_test_ss)

# ic(y_train_sk.shape, y_test_sk.shape)
# ic(y_train_ss.shape, y_test_ss.shape)


#2 Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Dropout, MaxPooling1D, Flatten
input1 = Input(shape=(4, 1))
xx = LSTM(64, activation='relu')(input1)
xx = Dense(32, activation='relu')(xx)
xx = Dense(8, activation='relu')(xx)
output1 = Dense(1)(xx)
model = Model(inputs=input1, outputs=output1)

input2 = Input(shape=(4, 1))
xx = LSTM(64, activation='relu')(input2)
xx = Dense(32, activation='relu')(xx)
xx = Dense(8, activation='relu')(xx)
output2 = Dense(1)(xx)
model = Model(inputs=input2, outputs=output2)

from tensorflow.keras.layers import Concatenate
merge1 = Concatenate(axis=1)([output1, output2])
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)

# model.summary()


#3 Compile / Train
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)

#########################################################################
import datetime
date = datetime.datetime.now()
date_time = date.strftime("%y%m%d_%H%M")

filepath = './samsung/_save/ModelCheckPoint/'
filename = '.{epoch:04d}-{val_loss:.4f}.hdf5'
modelpath = "".join([filepath, "SAMSUNG_", date_time, "_", filename])
#########################################################################

cp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=modelpath)

model.save('./samsung/_save/ModelCheckPoint/SAMSUNG_MCP.h5')

model.fit([x_train_sk, x_train_ss], y_train_ss, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es, cp])


#4 Evaluate / Predict
from sklearn.metrics import r2_score
loss = model.evaluate([x_test_sk, x_test_ss], y_test_ss)   # evaluate -> return loss, metrics
pred = model.predict([x_test_sk, x_test_ss])
r2 = r2_score(y_test_ss, pred)

ic(pred[:1])
ic('loss[mse] / metrics[acc]')
ic(loss)
ic(r2)

'''
ic| pred[:1]: array([[81262.59]], dtype=float32)
ic| 'loss[mse] / metrics[acc]'
ic| loss: [351261.5, 0.0]
ic| r2: 0.998695234433801
'''

# 5. Visualization
# plt.figure(figsize=(16,9))
# sns.lineplot(y=stock_ss['Close'], x=stock_ss.index)
# plt.xlabel('time')
# plt.ylabel('price')
# plt.show()