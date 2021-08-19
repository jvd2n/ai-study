from imblearn.over_sampling import SMOTE
from scipy.sparse import data
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

datasets = pd.read_csv('./_data/winequality-white.csv', index_col=None, header=0, sep=';').values   # values: to_numpy values

x = datasets[:, :11]
y = datasets[:, 11]
print(x.shape, y.shape) # (4898, 11) (4898,)
print(pd.Series(y).value_counts())
# 6.0    2198
# 5.0    1457
# 7.0     880
# 8.0     175
# 4.0     163
# 3.0      20
# 9.0       5
print(y[1])
############## 라벨 대통합!!? ##############
print("################################################")
for i, j in enumerate(y):
    if j == 9:
        y[i] = 2
    elif j == 8:
        y[i] = 2
    elif j == 7:
        y[i] = 1
    elif j == 6:
        y[i] = 1
    elif j == 5:
        y[i] = 0
    elif j == 4:
        y[i] = 0
    elif j == 3:
        y[i] = 0
    else:
        pass
# y = np.where(y == 9, 8, y)
print(pd.Series(y).value_counts())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=66, stratify=y
)

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print('model.score: ', score)   # model.score:  0.643265306122449


print('################### SMOTE ###################')
# ValueError: Expected n_neighbors <= n_samples,  but n_samples = 4, n_neighbors = 6
smote = SMOTE(random_state=77) #, k_neighbors=100)
start_time = time.time()
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)
end_time = time.time() - start_time

print('smote 전: ', x_train.shape, y_train.shape)
print('smote 후: ', x_smote_train.shape, y_smote_train.shape)
print('smote 전 레이블 값 분포: \n', pd.Series(y_train).value_counts())
print('smote 후 레이블 값 분포: \n', pd.Series(y_smote_train).value_counts())
print('SMOTE 경과 시간: ', end_time)
model2 = XGBClassifier(n_jobs=-1)
model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score = model2.score(x_test, y_test)
print('model2.score', score)