from icecream import ic
import math, time, datetime, warnings, numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes


warnings.filterwarnings('ignore')

#1 Data
datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
ic(x.shape, y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=66
)

#2 Scaling / Modeling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

model = make_pipeline(MinMaxScaler(), RandomForestClassifier())

#3 Compile / Train
model.fit(x_train, y_train)

#4 Evaluate / Predict
start_time = time.time()
model.fit(x_train, y_train)

#4Evaluate,Predict
# print("최적의 매개변수는 ", model.best_estimator_)
# print("best_params_ : ", model.best_params_)
# print("best_score_ : ", model.best_score_)

print("model.score : ", model.score(x_test, y_test))

y_pred = model.predict(x_test)
# print("accuracy_score : ", accuracy_score(y_test, y_pred))
print("accuracy_score : ", accuracy_score(y_test, y_pred))

print('Elapsed_Time : ', str(datetime.timedelta(seconds=math.ceil(time.time()-start_time))))

'''
ic| x.shape: (178, 13), y.shape: (178,)
model.score :  1.0
accuracy_score :  1.0
Elapsed_Time :  0:00:01
'''