# m13으로 만든 0.999 이상의 n_component를 사용하여 xgb 모델을 만들 것 (디폴트)
# mnist DNN보다 성능이 좋도록
# RandomSearch로도 해볼 것

import time, datetime, warnings, numpy as np
from icecream import ic
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
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
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)  # (60000,) (10000,)
x = np.append(x_train, x_test, axis=0)
print(x.shape)  # (70000, 28, 28)

# pca를 통해 0.95 이상인 n_components가 몇 개?
# Modeling
# Tensorflow DNN으로 구성하고 기존 DNN과 비교

N_COMPONENTS = 486
pca = PCA(n_components=N_COMPONENTS)   # 컬럼을 n_components개로 압축
# x_train = pca.fit_transform(x_train.reshape(-1, 28*28))
# x_test = pca.fit_transform(x_test.reshape(-1, 28*28))
x = pca.fit_transform(x.reshape(-1, 28*28))
print(x.shape)
x_train = x[:60000]
x_test = x[-10000:]
print(x_train.shape, x_test.shape)

N_SPLITS = 5
kfold = KFold(
    n_splits=N_SPLITS, # train과 test를 5개로 분할하여 다섯 번 훈련함
    shuffle=True,
    random_state=66,
)

PARAMETERS = [
    {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.3, 0.001, 0.01], 'max_depth': [4, 5, 6]},
    {'n_estimators': [90, 100, 110], 'learning_rate': [0.1, 0.001, 0.01], 'max_depth': [4, 5, 6], 'colsample_bytree': [0.6, 0.9, 1]},
    {'n_estimators': [90, 110], 'learning_rate': [0.1, 0.001, 0.5], 'max_depth': [4, 5, 6], 'colsample_bytree': [0.6, 0.9, 1], 'colsample_bylevel': [0.6, 0.7, 0.9]}
]
N_JOBS = -1

#2 Modeling
#2Modeling        
model = GridSearchCV(XGBRegressor(), PARAMETERS, cv=kfold, verbose=1)   # cv가 5회(N_SPLITS) # cross_validation?
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