import warnings
import numpy as np
from icecream import ic
from sklearn.datasets import load_diabetes

warnings.filterwarnings('ignore')

datasets = load_diabetes()
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
    KNeighborsRegressor(), 
    DecisionTreeRegressor(), 
    RandomForestRegressor(),
]
# ic(type(model))

# 3. 컴파일, 훈련

# 4. 평가, 예측
for model in models:
    scores = cross_val_score(model, x, y, cv=kfold)
    print(model, '// Acc : ', scores, ' /  AVG(Acc) : ', round(np.mean(scores),4))   # fit ~ score


'''
KNeighborsRegressor() // Acc :  [0.39683913 0.32569788 0.43311217 0.32635899 0.35466969]  /  AVG(Acc) :  0.3673
DecisionTreeRegressor() // Acc :  [-0.23853387 -0.1738592  -0.17189106  0.01748475  0.0067619 ]  /  AVG(Acc) :  -0.112
RandomForestRegressor() // Acc :  [0.3720317  0.48142879 0.48923475 0.42326844 0.42376722]  /  AVG(Acc) :  0.4379
'''