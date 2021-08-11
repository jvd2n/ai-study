import time, datetime, warnings, numpy as np
from icecream import ic
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score, r2_score


warnings.filterwarnings('ignore')

#1 Data
datasets = load_breast_cancer()
# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66
)

N_SPLITS = 5
kfold = KFold(
    n_splits=N_SPLITS, # train과 test를 5개로 분할하여 다섯 번 훈련함
    shuffle=True,
    random_state=66,
)

# PARAMETERS = [
#     {'n_jobs': [-1], 'n_estimators': [100, 200], 'max_depth': [6, 8, 10], 'min_samples_leaf': [5, 7, 10]},
#     {'n_jobs': [-1], 'max_depth': [5, 6, 7, 9], 'min_samples_leaf': [3, 6, 9, 11], 'min_samples_split': [3, 4, 5]},
#     {'n_jobs': [-1], 'min_samples_leaf': [3, 5, 7], 'min_samples_split': [2, 3, 5, 10]},
#     {'n_jobs': [-1], 'min_samples_split': [2, 3, 5, 10]},
# ]

PARAMETERS = [
    {'randomforestclassifier__min_samples_leaf': [3, 5, 7], 'randomforestclassifier__max_depth': [2, 3, 5, 10]},
    {'randomforestclassifier__min_samples_split': [6, 8, 10]},
]

#2 Modeling
pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
model = GridSearchCV(pipe, PARAMETERS, cv=kfold, verbose=1)
# model = RandomizedSearchCV(pipe, PARAMETERS, cv=kfold, verbose=1)

#3 Train
start_time = time.time()
model.fit(x_train, y_train)

#4 Evaluate,Predict
print("최적의 매개변수는 ", model.best_estimator_)
print("best_params_ : ", model.best_params_)
print("best_score_ : ", model.best_score_)

print("model.score : ", model.score(x_test, y_test))

y_pred = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_pred))
# print("r2_score : ", r2_score(y_test, y_pred))

print('Elapsed_Time : ', str(datetime.timedelta(seconds=round(time.time()-start_time, 0))))

'''
GridSearchCV
ic| x.shape: (569, 30), y.shape: (569,)
Fitting 5 folds for each of 127 candidates, totalling 635 fits
최적의 매개변수는  RandomForestClassifier(n_jobs=4)
best_params_ :  {'min_samples_split': 2, 'n_jobs': 4}
best_score_ :  0.9648351648351647
model.score :  0.956140350877193
accuracy_score :  0.956140350877193
Elapsed_Time :  0:01:51
'''
'''
ic| x.shape: (569, 30), y.shape: (569,)
Fitting 5 folds for each of 10 candidates, totalling 50 fits
최적의 매개변수는  RandomForestClassifier(max_depth=10, min_samples_leaf=3, n_estimators=200)
best_params_ :  {'n_estimators': 200, 'min_samples_leaf': 3, 'max_depth': 10}
best_score_ :  0.9626373626373625
model.score :  0.956140350877193
accuracy_score :  0.956140350877193
Elapsed_Time :  0:00:09
'''
'''
pipeline+GridSearchCV
ic| x.shape: (569, 30), y.shape: (569,)
Fitting 5 folds for each of 15 candidates, totalling 75 fits
최적의 매개변수는  Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('randomforestclassifier',
                 RandomForestClassifier(max_depth=10, min_samples_leaf=7))])
best_params_ :  {'randomforestclassifier__max_depth': 10, 'randomforestclassifier__min_samples_leaf': 7}
best_score_ :  0.9604395604395604
model.score :  0.956140350877193
accuracy_score :  0.956140350877193
Elapsed_Time :  0:00:09
'''