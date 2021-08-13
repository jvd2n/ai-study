import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1 Data
datasets = load_boston()
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape) # (506, 13) (506,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2 Model
model = XGBRegressor(n_estimators=200, learning_rate=0.05, n_jobs=1)

#3 Train
model.fit(
    x_train, y_train, 
    verbose=1,
    eval_set=[(x_train, y_train), (x_test, y_test)], # validation_data
    eval_metric='rmse',    # ['rmse', 'mae', 'logloss'],
    early_stopping_rounds=10,
)

#4 Evaluate
results = model.score(x_test, y_test)
print('results : ', results)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 : ', r2)

print('**********************************************************************')
hist = model.evals_result()
ic(hist)
print(type(hist))
print(hist['validation_0'])
for key, val in hist['validation_0'].items():
    print(key, val)
# 그래프
# plt.
'''
model = XGBRegressor(n_estimators=10000)
results :  0.9220477087047083
r2 :  0.9220477087047083

model = XGBRegressor(n_estimators=200, learning_rate=0.1, n_jobs=1)
results :  0.9314359074613965
r2 :  0.9314359074613965

model = XGBRegressor(n_estimators=300, learning_rate=0.05, n_jobs=1)
results :  0.9336635688400665
r2 :  0.9336635688400665

model = XGBRegressor(n_estimators=100, learning_rate=0.05, n_jobs=1)
results :  0.9354279880084962
r2 :  0.9354279880084962
'''