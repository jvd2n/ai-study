import time, datetime, warnings, numpy as np
from icecream import ic
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score


warnings.filterwarnings('ignore')

datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
ic(x.shape, y.shape)

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
    {'n_estimators': [100, 200], 'max_depth': [6, 8, 10, 12], 'min_samples_leaf': [3, 5, 7, 10]},
    {'max_depth': [5, 6, 7, 8], 'min_samples_leaf': [4, 6, 9, 11], 'min_samples_split': [2, 3, 4, 5]},
    {'min_samples_leaf': [3, 5, 7, 10], 'min_samples_split': [2, 3, 5, 7]},
    {'min_samples_split': [2, 3, 5, 10], 'n_jobs': [-1, 2, 4]},
    {'n_jobs': [-1, 4, 6]},
]

#2Modeling        
model = GridSearchCV(RandomForestClassifier(), PARAMETERS, cv=kfold, verbose=1)   # cv가 5회(N_SPLITS) # cross_validation?
# model = SVC()

#3Train
start_time = time.time()
model.fit(x_train, y_train)

#4Evaluate,Predict
print("최적의 매개변수는 ", model.best_estimator_)
print("best_params_ : ", model.best_params_)
print("best_score_ : ", model.best_score_)

print("model.score : ", model.score(x_test, y_test))

y_pred = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_pred))

print('Elapsed_Time : ', str(datetime.timedelta(seconds=round(time.time()-start_time, 0))))

'''
ic| x.shape: (150, 4), y.shape: (150,)
Fitting 5 folds for each of 127 candidates, totalling 635 fits
최적의 매개변수는  RandomForestClassifier(max_depth=5, min_samples_leaf=9, min_samples_split=5)
best_params_ :  {'max_depth': 5, 'min_samples_leaf': 9, 'min_samples_split': 5}
best_score_ :  0.9583333333333334
model.score :  1.0
accuracy_score :  1.0
Elapsed_Time :  0:01:37
'''