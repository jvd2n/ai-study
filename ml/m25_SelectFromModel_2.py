#1 그리드서치 혹은 랜덤서치로 튜닝한 모델 구성하여 최적의 R2값과 피처임포턴스 구할 것

#2 위 스레드 값으로 SFM 돌려서 최적의 피처 개수 구할 것

#3 위 피처 개수로 피처 개수를 조정한 뒤 다시 랜덤서치, 그리드서치하여 최적의 R2 구할 것

#1과 3 값 비교 후 0.47 이상으로 만들 것

from xgboost import XGBRegressor
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
# datasets = load_boston()
# x = datasets.data
# y = datasets.target
x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=148
)

#2 Modeling
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV

PARAMETERS = [
    {'n_estimators': [100, 200], 'max_depth': [6, 8, 10, 12]},
    {'max_depth': [5, 6, 7, 8]},
    {'n_jobs': [-1, 2, 4, 6]},
]

model = RandomizedSearchCV(
    XGBRegressor(), 
    PARAMETERS,
    verbose=1
)

# model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
            #  colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
            #  importance_type='gain', interaction_constraints='',
            #  learning_rate=0.300000012, max_delta_step=0, max_depth=6,
            #  min_child_weight=1, missing=nan, monotone_constraints='()',
            #  n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
            #  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
            #  tree_method='exact', validate_parameters=1, verbosity=None)

#3 Train
model.fit(x_train, y_train)

#4 Evaluate/Predict
score = model.score(x_test, y_test)
print('model.score : ', score)
print('Best estimator : ', model.best_estimator_)
print('Best score  :', model.best_score_)
y_pred = model.predict(x_test)
print('r2_score: ', r2_score(y_test, y_pred))

'''
Grid
model.score :  0.23802704693460175
Best estimator :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.300000012, max_delta_step=0, max_depth=6,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
Best score  : 0.4060698714523321

Rand
model.score :  0.23802704693460175
Best estimator :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.300000012, max_delta_step=0, max_depth=6,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=100, n_jobs=2, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
Best score  : 0.4060698714523321
'''


# aaa = model.feature_importances_
# print(aaa)

# thresholds = np.sort(model.feature_importances_)
# print(thresholds)

# for thresh in thresholds:
#     print(thresh)
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
#     # print(selection)
    
#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)
    
#     print(select_x_train.shape, select_x_test.shape)
    
#     selection_model = XGBRegressor(n_jobs=-1)
#     selection_model.fit(select_x_train, y_train)
    
#     y_pred = selection_model.predict(select_x_test)

#     score = r2_score(y_test, y_pred)
    
#     print('Thresh=%.3f, n=%d, R2 %.2f%%' %(thresh, select_x_train.shape[1], score*100))