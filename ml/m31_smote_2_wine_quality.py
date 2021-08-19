from imblearn.over_sampling import SMOTE
from scipy.sparse import data
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

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=66, stratify=y
)
print(pd.Series(y_train).value_counts())
# 6.0    1648
# 5.0    1093
# 7.0     660
# 8.0     131
# 4.0     122
# 3.0      15
# 9.0       4

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print('model.score: ', score)   # model.score:  0.643265306122449


print('################### SMOTE ###################')
# ValueError: Expected n_neighbors <= n_samples,  but n_samples = 4, n_neighbors = 6
smote = SMOTE(random_state=66, k_neighbors=3)
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)

# print(pd.Series(y_smote_train).value_counts())
# # 2    53
# # 1    53
# # 0    53
# print(x_smote_train.shape, y_smote_train.shape) # (159, 13) (159,)

print('smote 전: ', x_train.shape, y_train.shape)
print('smote 후: ', x_smote_train.shape, y_smote_train.shape)
print('smote 전 레이블 값 분포: \n', pd.Series(y_train).value_counts())
print('smote 후 레이블 값 분포: \n', pd.Series(y_smote_train).value_counts())

model2 = XGBClassifier(n_jobs=-1)
model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score = model2.score(x_test, y_test)
print('model2.score', score)    # model2.score 0.6318367346938776


############## 라벨 대통합!!? ##############
