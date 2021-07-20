from icecream import ic
import numpy as np
from sklearn.datasets import load_breast_cancer

datasets = load_breast_cancer()

ic(datasets.DESCR)
ic(datasets.feature_names)

x = datasets.data
y = datasets.target

ic(x.shape, y.shape)    # (569, 30) (569,)

ic(y[:20])
ic(np.unique(y))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)