import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

#1 data
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

#2 model
model = SVC()

#3 train
model.fit(x_data, y_data)

#4 evaluate, predict
y_pred = model.predict(x_data)
print(x_data, '의 예측 결과 : ', y_pred)

results = model.score(x_data, y_data)
print('model.score : ', results)

acc = accuracy_score(y_data, y_pred)
print('acc_score : ', acc)