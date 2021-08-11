import warnings
import numpy as np
from icecream import ic
from sklearn.datasets import load_iris
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

warnings.filterwarnings('ignore')

datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

ic(x.shape, y.shape)    # (150, 4) (150,)

from sklearn.model_selection import train_test_split, KFold, cross_val_score
# cross_val_score : 교차검증한 값을 뽑아내므로, fit과 score를 한 번에 가져옴
N_SPLITS = 5

kfold = KFold(
    n_splits=N_SPLITS, # train과 test를 5개로 분할하여 다섯 번 훈련함
    shuffle=True,
    random_state=66,
)

#2. 모델 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류는 Classifier, 회귀는 Regressor
from sklearn.linear_model import LogisticRegression # 이름과는 달리 (논리적)회귀 모델이 아닌 분류 모델임
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # DecisionTree의 확장형 구조 (Forest)

# model = LinearSVC()               # Acc :  [0.96666667 0.96666667 1.         0.9        1.        ]  /  AVG(Acc) :  0.9667
# model = SVC()                     # Acc :  [0.96666667 0.96666667 1.         0.93333333 0.96666667]  /  AVG(Acc) :  0.9667
model = KNeighborsClassifier()    # Acc :  [0.96666667 0.96666667 1.         0.9        0.96666667]  /  AVG(Acc) :  0.96
# model = KNeighborsRegressor()     # Acc :  [0.93758389 0.972      0.9942029  0.85572519 0.97487923]  /  AVG(Acc) :  0.9469
# model = LogisticRegression()      # Acc :  [1.         0.96666667 1.         0.9        0.96666667]  /  AVG(Acc) :  0.9667
# model = DecisionTreeClassifier()  # Acc :  [0.93333333 0.96666667 1.         0.9        0.93333333]  /  AVG(Acc) :  0.9467
# model = DecisionTreeRegressor()   # Acc :  [0.89932886 0.95       1.         0.82824427 0.90338164]  /  AVG(Acc) :  0.9162
# model = RandomForestClassifier()  # Acc :  [0.96666667 0.96666667 1.         0.9        0.96666667]  /  AVG(Acc) :  0.96
# model = RandomForestRegressor()   # Acc :  [0.95207047 0.938505   0.99977778 0.87207634 0.9580628 ]  /  AVG(Acc) :  0.9441

# 3. 컴파일, 훈련

# 4. 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold)
print('Acc : ', scores, ' /  AVG(Acc) : ', round(np.mean(scores),4))   # fit ~ score
# [0.96666667 0.96666667 1.         0.9        1.        ]     => accuracy value

