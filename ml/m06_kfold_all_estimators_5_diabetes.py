import warnings
import numpy as np
from icecream import ic
from sklearn.datasets import load_diabetes
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score

warnings.filterwarnings('ignore')

datasets = load_diabetes()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

#2. 모델 구성
# from sklearn.utils.testing import all_estimators

allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
# ic(allAlgorithms)
ic(len(allAlgorithms))  # 41

kfold = KFold(n_splits=5, shuffle=True, random_state=66)
for i, (name, algorithm) in enumerate(allAlgorithms):
    try:
        model = algorithm()
        
        scores = cross_val_score(model, x, y, cv=kfold)
        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        # acc = accuracy_score(y_test, y_pred)
        # print('[', i+1 ,'] ', name, '의 정답률 : ', acc)
        print('[', i+1 ,'] ', name, scores, ' [ ***AVG*** ', round(np.mean(scores), 4), ' ]')
    except:
        # continue
        print('[', i+1 ,'] ', name, '은 예외 처리 되었습니다.')
