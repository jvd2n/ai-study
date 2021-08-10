# from sklearn.utils.testing import all_estimators
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

from icecream import ic
import numpy as np
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings('ignore')

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    shuffle=True,
    random_state=66
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성

allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
# ic(allAlgorithms)
ic(len(allAlgorithms))  # 41
for i, (name, algorithm) in enumerate(allAlgorithms):
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print('[', i ,'] ', name, '의 정답률 : ', acc)
    except:
        # continue
        print('[', i ,'] ', name, '은 예외 처리 되었습니다.')

'''
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = LogisticRegression()

# 3. 컴파일, 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('model.score : ', results)
# loss = model.evaluate(x_test, y_test)   # evaluate -> return loss, metrics
# print(f'loss: {loss[0]}')
# print(f'accuracy: {loss[1]}')

from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score : ', acc)

print('================= PREDICT =================')
print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)
'''