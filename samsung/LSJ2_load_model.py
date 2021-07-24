from icecream import ic 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#1 Data / Preprocessing
stock_sk = pd.read_excel('./_data/SK주가 20210721.xls', index_col=None, header=0, usecols=[0,1,2,3,4,10])
stock_ss = pd.read_excel('./_data/삼성전자 주가 20210721.xls', index_col=None, header=0, usecols=[0,1,2,3,4,10])

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
stock_ss = stock_ss.sort_index(ascending=True)
stock_sk = stock_sk.sort_index(ascending=True)
ic(stock_ss, stock_sk)

from sklearn.preprocessing import MinMaxScaler, RobustScaler
scaler = MinMaxScaler()
scale_cols = ['High', 'Low', 'Close', 'Volume']
scaled_ss = scaler.fit_transform(stock_ss[scale_cols])
scaled_sk = scaler.fit_transform(stock_sk[scale_cols])

df_ss = pd.DataFrame(scaled_ss, columns=scale_cols)
df_sk = pd.DataFrame(scaled_sk, columns=scale_cols)
df_ss_close = stock_ss['Open'].reset_index()
df_ss_close = df_ss_close['Open']
df_sk_close = stock_sk['Open'].reset_index()
df_sk_close = df_sk_close['Open']

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
    for i in range(len(data) - window_size - 2):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size+2]))
    return np.array(feature_list), np.array(label_list)

feature_cols = ['High', 'Low', 'Close', 'Volume']
label_cols = ['Open']

############# SAMSUNG DATA #################
train_feature_ss = train_ss[feature_cols]
train_label_ss = train_ss[label_cols]
test_feature_ss = test_ss[feature_cols]
test_label_ss = test_ss[label_cols]

train_feature_ss, train_label_ss = make_dataset(train_feature_ss, train_label_ss, 20)
test_feature_ss, test_label_ss = make_dataset(test_feature_ss, test_label_ss, 20)
ic(train_feature_ss.shape, train_label_ss.shape)
ic(test_feature_ss.shape, test_label_ss.shape)

############# SK DATA #################
train_feature_sk = train_sk[feature_cols]
train_label_sk = train_sk[label_cols]
test_feature_sk = test_sk[feature_cols]
test_label_sk = test_sk[label_cols]

train_feature_sk, train_label_sk = make_dataset(train_feature_sk, train_label_sk, 20)
test_feature_sk, test_label_sk = make_dataset(test_feature_sk, test_label_sk, 20)

ic(train_feature_sk.shape, train_label_sk.shape)
ic(test_feature_sk.shape, test_label_sk.shape)

ic(test_feature_ss, test_label_ss)

from keras.models import load_model
# model = load_model('D:\study\samsung\_save\ModelCheckPoint\SAMSUNG2_210724_2114_.0054-35199976.0000.hdf5')
# model = load_model('D:\study\samsung\_save\ModelCheckPoint\SAMSUNG2_210724_2121_.0027-35105820.0000.hdf5')
model = load_model('D:\study\samsung\_save\ModelCheckPoint\SAMSUNG2_210724_2309_.0051-40121096.0000.hdf5')

#4 Evaluate / Predict
loss = model.evaluate([test_feature_ss, test_feature_sk], test_label_ss)   # evaluate -> return loss, metrics
pred = model.predict([test_feature_ss, test_feature_sk])

ic(pred[-1])
# ic(pred.shape)
ic(loss)

'''
ic| pred[-1]: array([79628.04], dtype=float32)
ic| loss: 4269286.0
'''
# 5. Visualization
# plt.figure(figsize=(16,9))
# sns.lineplot(y=stock_ss['Close'], x=stock_ss.index)
# plt.xlabel('time')
# plt.ylabel('price')
# plt.show()