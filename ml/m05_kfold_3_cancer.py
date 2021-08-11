import warnings
import numpy as np
from icecream import ic
from sklearn.datasets import load_breast_cancer

warnings.filterwarnings('ignore')

datasets = load_breast_cancer()
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

'''
LinearSVC() // Acc :  [0.8245614  0.93859649 0.9122807  0.93859649 0.9380531 ]  /  AVG(Acc) :  0.9104
SVC() // Acc :  [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177]  /  AVG(Acc) :  0.921
KNeighborsClassifier() // Acc :  [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221]  /  AVG(Acc) :  0.928
LogisticRegression() // Acc :  [0.93859649 0.95614035 0.88596491 0.95614035 0.96460177]  /  AVG(Acc) :  0.9403
DecisionTreeClassifier() // Acc :  [0.92105263 0.9122807  0.92105263 0.89473684 0.92920354]  /  AVG(Acc) :  0.9157
RandomForestClassifier() // Acc :  [0.96491228 0.96491228 0.95614035 0.94736842 0.98230088]  /  AVG(Acc) :  0.9631
'''