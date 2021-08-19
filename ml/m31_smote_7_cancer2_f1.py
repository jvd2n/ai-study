from imblearn.over_sampling import SMOTE
from scipy.sparse import data
from sklearn.datasets import load_wine, load_breast_cancer
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time
import warnings

from xgboost.training import train
warnings.filterwarnings('ignore')

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)
print(y)
print(pd.Series(y).value_counts())
y = np.sort(y)
print(y)
# 1    357
# 0    212

x = x[:-100]
y = y[:-100]
print(x.shape, y.shape) # (569, 30) (569,)
print(pd.Series(y).value_counts())
# 1    257
# 0    212

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=66,
    # stratify=y
)

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print('model.score: ', score)   # model.score:  0.643265306122449

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print('f1_score: ', f1)