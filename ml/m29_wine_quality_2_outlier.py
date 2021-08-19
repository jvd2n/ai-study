# 아웃라이어 확인
from icecream import ic 
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from xgboost import XGBClassifier

#1 Data
datasets = pd.read_csv('./_data/winequality-white.csv', index_col=None, header=0, sep=';')

print(datasets.head())
print(datasets.shape)   # (4898, 12)
print(datasets.describe())
print(datasets.columns)
data_header = datasets.columns

datasets = datasets.values  # to_numpy transform
# print(type(datasets))   # <class 'numpy.ndarray'>
# print(datasets.shape)   # (4898, 12)
# ic(datasets)
x = datasets[:, :11]
y = datasets[:, 11]

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66
)

ic(x_train)
ic(x_train.shape[1])
ic(x_train[:, 0])
# ic(x_train[:, 0][184])

outliers_idx = []
def outliers(data_out):
    for i in range(0, data_out.shape[1]):
        data_out = np.delete(data_out, np.where(data_out == [0, 0, 0]), axis=0)
        quartile_1, q2, quartile_3 = np.percentile(data_out[:, i], [25, 50, 75])
        print('\n', '*'*20, i+1, '열 /', data_header[i], '*'*20)
        print('1사분위: ', quartile_1)
        print('q2 :', q2)
        print('3사분위: ', quartile_3)
        iqr = quartile_3 - quartile_1   # IQR(Inter Quartile Range, 사분범위)
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        ic(lower_bound, upper_bound)
        outliers = np.where((data_out[:, i] > upper_bound) | (data_out[:, i] < lower_bound))
        outliers_idx.append(outliers)
        print(f'{i+1}번 째 열({data_header[i]})의 이상치 인덱스 값:\n', outliers)
    return
outliers_loc = outliers(x_train)
# ic(outliers_idx)

ol_stack = np.hstack(outliers_idx)
ol_stack_sort = np.sort(ol_stack)
ol_stack_sort_unique = np.unique(ol_stack_sort)
ic(ol_stack_sort_unique)
print('이상치가 존재하는 열의 개수 : ', ol_stack_sort_unique.shape[0], '개')
##### 아웃라이어의 개수를 count 하도록 기능 추가
# print(len(np.unique(outliers_loc[0])))
# ic(np.unique(outliers_loc[0][:50]))


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