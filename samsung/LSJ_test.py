from icecream import ic 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#1 Data / Preprocessing
stock_sk = pd.read_excel('./_data/SK주가 20210721.xls', 
                         index_col=None, header=0, usecols=[0,1,2,3,4,10])
stock_ss = pd.read_excel('./_data/삼성전자 주가 20210721.xls', 
                         index_col=None, header=0, usecols=[0,1,2,3,4,10])

stock_sk.columns = ['date','Open','High','Low','Close','Volume']
stock_ss.columns = ['date','Open','High','Low','Close','Volume']
stock_sk = stock_sk.iloc[:2601,:]
stock_ss = stock_ss.iloc[:2601,:]

ic(stock_sk.shape, stock_ss.shape)   # (2601, 5) , (2601, 5)
# stock_sk['date'] = stock_sk['date'].str.replace("/", "")
stock_sk['date'] = pd.to_datetime(stock_sk['date'])
stock_ss['date'] = pd.to_datetime(stock_ss['date'])
stock_sk.set_index('date', inplace=True)
stock_ss.set_index('date', inplace=True)
# ic(stock_sk['date'])
# ic(stock_ss['date'])
stock_ss = stock_ss.sort_index(ascending=True)
stock_sk = stock_sk.sort_index(ascending=True)
# ic(stock_ss.info)
# ic(stock_sk.info)
ic(stock_ss, stock_sk)

from sklearn.preprocessing import MinMaxScaler, RobustScaler
scaler = MinMaxScaler()
scale_cols = ['Open', 'High', 'Low', 'Volume']
scaled_ss = scaler.fit_transform(stock_ss[scale_cols])
scaled_sk = scaler.fit_transform(stock_sk[scale_cols])
# ic(scaled_ss)
# ic(scaled_sk)
df_ss = pd.DataFrame(scaled_ss, columns=scale_cols)
df_sk = pd.DataFrame(scaled_sk, columns=scale_cols)
df_ss_close = stock_ss['Close'].reset_index()
df_ss_close = df_ss_close['Close']
df_sk_close = stock_sk['Close'].reset_index()
df_sk_close = df_sk_close['Close']

df_scaled_ss = pd.concat([df_ss, df_ss_close], axis=1)
df_scaled_sk = pd.concat([df_sk, df_sk_close], axis=1)
ic(df_scaled_ss, df_scaled_sk)

train_ss = df_scaled_ss[:-100]
test_ss = df_scaled_ss[-100:]
train_sk = df_scaled_sk[:-100]
test_sk = df_scaled_sk[-100:]
ic(train_ss, test_ss, train_sk, test_sk)

def make_dataset(data, label, window_size):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)

feature_cols = ['Open', 'High', 'Low', 'Volume']
label_cols = ['Close']

############# SAMSUNG DATA #################
train_feature_ss = train_ss[feature_cols]
train_label_ss = train_ss[label_cols]
test_feature_ss = test_ss[feature_cols]
test_label_ss = test_ss[label_cols]

train_feature_ss, train_label_ss = make_dataset(train_feature_ss, train_label_ss, 20)
# ic(train_feature_ss, train_label_ss)
ic(train_feature_ss.shape, train_label_ss.shape)

test_feature_ss, test_label_ss = make_dataset(test_feature_ss, test_label_ss, 20)
# ic(test_feature_ss, test_label_ss)
ic(test_feature_ss.shape, test_label_ss.shape)

from sklearn.model_selection import train_test_split
x_train_ss, x_valid_ss, y_train_ss, y_valid_ss = train_test_split(train_feature_ss, train_label_ss, test_size=0.2)
ic(x_train_ss.shape, x_valid_ss.shape)
ic(y_train_ss.shape, y_valid_ss.shape)

############# SK DATA #################
train_feature_sk = train_sk[feature_cols]
train_label_sk = train_sk[label_cols]
test_feature_sk = test_sk[feature_cols]
test_label_sk = test_sk[label_cols]

train_feature_sk, train_label_sk = make_dataset(train_feature_sk, train_label_sk, 20)
# ic(train_feature_sk, train_label_sk)
ic(train_feature_sk.shape, train_label_sk.shape)

test_feature_sk, test_label_sk = make_dataset(test_feature_sk, test_label_sk, 20)
# ic(test_feature_sk, test_label_sk)
ic(test_feature_sk.shape, test_label_sk.shape)

from sklearn.model_selection import train_test_split
x_train_sk, x_valid_sk, y_train_sk, y_valid_sk = train_test_split(train_feature_sk, train_label_sk, test_size=0.2)

y_train_ss = y_train_ss.reshape(-1)
y_valid_ss = y_valid_ss.reshape(-1)
y_train_sk = y_train_sk.reshape(-1)
y_valid_sk = y_valid_sk.reshape(-1)

ic(x_train_ss.shape, x_valid_ss.shape)
ic(y_train_ss.shape, y_valid_ss.shape)
ic(x_train_sk.shape, x_valid_sk.shape)
ic(y_train_sk.shape, y_valid_sk.shape)


#2 Modeling
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM
model = Sequential()
model.add(LSTM(16, input_shape=(train_feature_ss.shape[1], train_feature_ss.shape[2]), activation='relu', return_sequences=False))
model.add(Dense(8))
model.add(Dense(1))

model.summary()


#3 Compile / Train
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)

# #########################################################################
# import datetime
# date = datetime.datetime.now()
# date_time = date.strftime("%y%m%d_%H%M")

# filepath = './samsung/_save/ModelCheckPoint/'
# filename = '.{epoch:04d}-{val_loss:.4f}.hdf5'
# modelpath = "".join([filepath, "SAMSUNG_", date_time, "_", filename])
# #########################################################################

# cp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=modelpath)

# model.save('./samsung/_save/ModelCheckPoint/SAMSUNG_MCP.h5')

model.fit(x_train_ss, y_train_ss, epochs=100, batch_size=32, validation_data=(x_valid_ss, y_valid_ss), callbacks=[es])


#4 Evaluate / Predict
from sklearn.metrics import r2_score
loss = model.evaluate(x_valid_ss, y_valid_ss)   # evaluate -> return loss, metrics
pred = model.predict(test_feature_ss)
# r2 = r2_score(y_valid_ss, pred)

ic(pred)
ic('loss[mse] / metrics[acc]')
ic(loss)
# ic(r2)


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