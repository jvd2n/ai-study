import time
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
model = XGBRegressor(
    n_estimators=10000, 
    learning_rate=0.01,
    tree_methd='gpu_hist',
    predictor='gpu_predictor',  # cpu_predictor 
    gpu_id=0,
    n_jobs=4,
)

#3 Train

start_time = time.time()
model.fit(
    x_train, y_train, 
    verbose=1,
    eval_set=[(x_train, y_train), (x_test, y_test)], # validation_data
    eval_metric='rmse',    # ['rmse', 'mae', 'logloss'],
    # early_stopping_rounds=10,
)
print("Elapsed_time : ", time.time() - start_time)
#4 Evaluate
