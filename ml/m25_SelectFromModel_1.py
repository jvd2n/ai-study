from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
# datasets = load_boston()
# x = datasets.data
# y = datasets.target
x, y = load_boston(return_X_y=True)
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66
)

#2 Modeling
model = XGBRegressor(n_jobs=8)

#3 Train
model.fit(x_train, y_train)

#4 Evaluate/Predict
score = model.score(x_test, y_test)
print('model.score : ', score)      # 0.9221

aaa = model.feature_importances_
print(aaa)

thresholds = np.sort(model.feature_importances_)
print(thresholds)

for thresh in thresholds:
    print(thresh)
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    # print(selection)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)
    
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    
    print('Thresh=%.3f, n=%d, R2 %.2f%%' %(thresh, select_x_train.shape[1], score*100))