# 피처 임포턴스가 전체 중요도에서 하위 20~25% 컬럼을 제거 하여 데이터셋을 재구성한 후
# 각 모델별로 결과 도출
# 기존 모델결과와 비교
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
from sklearn.model_selection import train_test_split

#1 Data
datasets = load_iris()
ic(datasets)
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=66
)

#2 Model
# model = DecisionTreeClassifier(max_depth=4)
model = RandomForestClassifier()

#3 Train
model.fit(x_train, y_train)

#4 Evaluate / Predict
acc = model.score(x_test, y_test)
print('acc: ', acc)

print(datasets.feature_names)
print(model.feature_importances_)
print(type(model.feature_importances_))
a = np.array(datasets.feature_names).reshape(1, datasets.data.shape[1])
b = model.feature_importances_.reshape(1, datasets.data.shape[1])
c = np.concatenate([a, b], axis=0)
d = np.transpose(c)
e = pd.DataFrame(d)
e.columns = ['Features', 'Feature Importances']
e = e.sort_values(by=['Feature Importances'], axis=0, ascending=False).head(math.ceil(datasets.data.shape[1] * 0.75))
print(e.values)
print(e.values[:,0])
print(e.values[:,1])

'''
print(datasets.feature_names)
print(datasets.data.shape[1])
print(type(datasets.feature_names))
print(type(model.feature_importances_))
print(np.sort(model.feature_importances_))

a = np.array(datasets.feature_names).reshape(1, datasets.data.shape[1])
b = model.feature_importances_.reshape(1, datasets.data.shape[1])
c = np.concatenate([a, b], axis=0)
d = np.transpose(c)
e = pd.DataFrame(d)
e.columns = ['Features', 'Feature Importances']
e = e.sort_values(by=['Feature Importances'], axis=0, ascending=False).head(math.ceil(datasets.data.shape[1] * 0.75))
print(e.values)
print(e.values[:,1])

# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#              align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)

# plot_feature_importances_dataset(model)
# plt.show()

def plot_feature_importances_dataset_2(model):
    a = np.array(datasets.feature_names).reshape(1, datasets.data.shape[1])
    b = model.feature_importances_.reshape(1, datasets.data.shape[1])
    c = np.concatenate([a, b], axis=0)
    d = np.transpose(c)
    e = pd.DataFrame(d)
    e.columns = ['Features', 'Feature Importances']
    e = e.sort_values(by=['Feature Importances'], axis=0, ascending=False).head(math.ceil(datasets.data.shape[1] * 0.75))
    
    n_features = datasets.data.shape[1] * 0.75
    plt.barh(np.arange(n_features), e.values[:,1],
             align='center')
    plt.yticks(np.arange(n_features), e.values[:,0])
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset_2(model)
plt.show()
'''