import warnings
import numpy as np
from icecream import ic
from sklearn.datasets import load_wine

warnings.filterwarnings('ignore')

datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

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

models = [
    LinearSVC(), 
    SVC(), 
    KNeighborsClassifier(), 
    LogisticRegression(), 
    DecisionTreeClassifier(), 
    RandomForestClassifier()
]
# ic(type(model))

# 3. 컴파일, 훈련

# 4. 평가, 예측
for model in models:
    scores = cross_val_score(model, x, y, cv=kfold)
    print(model, '// Acc : ', scores, ' /  AVG(Acc) : ', round(np.mean(scores),4))   # fit ~ score
    # [0.96666667 0.96666667 1.         0.9        1.        ]     => accuracy value

'''
LinearSVC() // Acc :  [0.61111111 0.75       0.86111111 0.77142857 0.85714286]  /  AVG(Acc) :  0.7702
SVC() // Acc :  [0.69444444 0.69444444 0.61111111 0.62857143 0.6       ]  /  AVG(Acc) :  0.6457
KNeighborsClassifier() // Acc :  [0.69444444 0.77777778 0.61111111 0.62857143 0.74285714]  /  AVG(Acc) :  0.691
LogisticRegression() // Acc :  [0.97222222 0.94444444 0.94444444 0.94285714 1.        ]  /  AVG(Acc) :  0.9608
DecisionTreeClassifier() // Acc :  [0.94444444 0.97222222 0.91666667 0.88571429 0.91428571]  /  AVG(Acc) :  0.9267
RandomForestClassifier() // Acc :  [1.         0.94444444 1.         0.97142857 1.        ]  /  AVG(Acc) :  0.9832
'''