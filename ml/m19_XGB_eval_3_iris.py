from xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import load_boston, load_diabetes, load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1 Data
datasets = load_iris()
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
model = XGBClassifier(n_estimators=100, learning_rate=0.05, n_jobs=1)    # n_jobs : 코어를 몇 개 쓸 것인가

#3 Train
model.fit(
    x_train, y_train, 
    verbose=1,
    eval_set=[(x_train, y_train), (x_test, y_test)], # validation_data
    # eval_metric='rmse',   # ['rmse', 'mae', 'logloss'],
)

#4 Evaluate
results = model.score(x_test, y_test)
print('results : ', results)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print('r2 : ', r2)
print('acc : ', acc)

'''
results :  0.9333333333333333
r2 :  0.8993288590604027
acc :  0.9333333333333333
'''