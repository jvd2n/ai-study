from icecream import ic
from sklearn.datasets import load_boston

datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape)
print(y.shape)

print(datasets.feature_names)
print(datasets.DESCR)

ic(x)

# 완료하시오!