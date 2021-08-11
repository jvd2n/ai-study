import warnings
import numpy as np
from icecream import ic
from sklearn.datasets import load_iris

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류는 Classifier, 회귀는 Regressor
from sklearn.linear_model import LogisticRegression # 이름과는 달리 (논리적)회귀 모델이 아닌 분류 모델임
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # DecisionTree의 확장형 구조 (Forest)

from sklearn.metrics import accuracy_score


warnings.filterwarnings('ignore')

datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

ic(x.shape, y.shape)    # (150, 4) (150,)

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
# cross_val_score : 교차검증한 값을 뽑아내므로, fit과 score를 한 번에 가져옴

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=66
)

N_SPLITS = 5
kfold = KFold(
    n_splits=N_SPLITS, # train과 test를 5개로 분할하여 다섯 번 훈련함
    shuffle=True,
    random_state=66,
)

PARAMETERS = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},                            # 4 x 5(cv) = 20
    {'C': [1, 10, 100], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},           # 3 x 1 x 2 x 5(cv) = 30
    {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}, # 4 x 1 x 2 x 5(cv) = 40
]                                                                               # 90회
#2Modeling        
model = GridSearchCV(SVC(), PARAMETERS, cv=kfold)   # cv가 5회(N_SPLITS) # cross_validation?
# model = SVC()

#3Train
model.fit(x_train, y_train)

#4Evaluate,Predict
print("최적의 매개변수 ", model.best_estimator_)
print("best_score_ : ", model.best_score_)

print("model.score : ", model.score(x_test, y_test))

y_pred = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_pred))