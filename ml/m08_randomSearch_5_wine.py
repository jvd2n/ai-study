import time, datetime, warnings, numpy as np
from icecream import ic
from sklearn.datasets import load_wine
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score


warnings.filterwarnings('ignore')

datasets = load_wine()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
ic(x.shape, y.shape)

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV

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
# model = GridSearchCV(RandomForestClassifier(), PARAMETERS, cv=kfold, verbose=1)   # cv가 5회(N_SPLITS) # cross_validation?
model = RandomizedSearchCV(RandomForestClassifier(), PARAMETERS, cv=kfold, verbose=1)
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
GridSearchCV
ic| x.shape: (178, 13), y.shape: (178,)
Fitting 5 folds for each of 127 candidates, totalling 635 fits
최적의 매개변수는  RandomForestClassifier(n_jobs=-1)
best_params_ :  {'min_samples_split': 2, 'n_jobs': -1}
best_score_ :  0.9788177339901478
model.score :  1.0
accuracy_score :  1.0
Elapsed_Time :  0:01:37
'''
'''
RandomizedSearchCV
ic| x.shape: (178, 13), y.shape: (178,)
Fitting 5 folds for each of 10 candidates, totalling 50 fits
최적의 매개변수는  RandomForestClassifier(min_samples_split=3, n_jobs=-1)
best_params_ :  {'n_jobs': -1, 'min_samples_split': 3}
best_score_ :  0.9716748768472907
model.score :  1.0
accuracy_score :  1.0
Elapsed_Time :  0:00:12
'''