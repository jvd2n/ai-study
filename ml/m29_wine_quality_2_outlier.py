# 아웃라이어 확인

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from xgboost import XGBClassifier

#1 Data
datasets = pd.read_csv('./_data/winequality-white.csv', index_col=None, header=0, sep=';')

print(datasets.head())
print(datasets.shape)   # (4898, 12)
print(datasets.describe())

# datasets = datasets.values  # to_numpy transform
# print(type(datasets))   # <class 'numpy.ndarray'>
# print(datasets.shape)   # (4898, 12)

# x = datasets[:, :11]
# y = datasets[:, 11]

# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, shuffle=True, random_state=66
# )

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print('1사분위: ', quartile_1)
    print('q2 :', q2)
    print('3사분위: ', quartile_3)
    iqr = quartile_3 - quartile_1   # IQR(Inter Quartile Range, 사분범위)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound) | (data_out < lower_bound))
# outliers_loc = outliers(x_train)
outliers_loc = datasets.apply(lambda x: outliers(x))
print('이상치의 위치: ', outliers_loc)
##### 아웃라이어의 개수를 count 하도록 기능 추가



# scaler = RobustScaler()
# # scaler.fit(x_train)
# # x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# #2 Model
# model = XGBClassifier(n_jobs=-1)

# #3 Train
# model.fit(x_train, y_train)

# #4 Eval, Pred
# score = model.score(x_test, y_test)

# print('accuracy: ', score)  # accuracy:  0.6826530612244898