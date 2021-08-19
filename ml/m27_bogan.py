# [1, np.nan, np.nan, 8, 10]

# 결측치 처리
#1 행 삭제
#2 0 넣기 (특정값)   -> [1, 0, 0, 8, 10]
#3 앞의 값           -> [1, 1, 1, 8, 10]
#4 뒤의 값           -> [1, 8, 8, 8, 10]
#5 중위값            -> [1, 4.5, 4.5, 8, 10]
#6 보간
#7 모델링 - predict
#8 부스트 계열은 결측치에 대해 자유롭다. 모델링: tree 계열, DT, RF, XG, LGBM ...

from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datastrs = ['8/13/2021', '8/14/2021', '8/15/2021', '8/16/2021', '8/17/2021']
dates = pd.to_datetime(datastrs)
print(dates)
print(type(dates))  # <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
print('*'*50)

ts = Series([1, np.nan, np.nan, 8, 10], index=dates)
print(ts)

ts_intp_linear = ts.interpolate()
print(ts_intp_linear)
