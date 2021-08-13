import numpy as np
from sklearn.datasets import load_diabetes, load_boston
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#1 Data
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,)

pca = PCA(n_components=10)   # 컬럼을 n_components개로 압축
x = pca.fit_transform(x)
# print(x)
print(x.shape)    # (442, 9)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)
print(cumsum)

print(np.argmax(cumsum >= 0.94) + 1)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()


# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, random_state=66, shuffle=True
# )

# #2 Modeling
# from xgboost import XGBRegressor
# model = XGBRegressor()

# #3 Train
# model.fit(x_train, y_train)

# #4 Evaluate / Predict
# results = model.score(x_test, y_test)
# print('결과 : ', results)

'''
xgb 결과 :  0.28428734040184866
pca 결과 :  0.3538690321164464
'''