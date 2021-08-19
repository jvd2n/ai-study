# 실습 / 다차원의 outliers 출력할 수 있도록
import numpy as np
from icecream import ic

aaa = np.array([[1, 2, 1000, 3, 4, 6, 7, 8, 90, 100, 1000],
                [1000, 2000, 3, 4000, 5000, 6000, 7000, 8, 5000, 20000, 1001]])
# (2, 10) -> (10, 2)
bbb = np.array([1, 2, -1000, 4, 5, 6, 7, 8, 90, 100, 500])
aaa = aaa.transpose()
# aaa = aaa.reshape(aaa.size, )
# # aaa = np.sort(aaa)
# print(aaa.shape)
# print(aaa)
# print(type(aaa))

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print('1사분위: ', quartile_1)
    print('q2 :', q2)
    print('3사분위: ', quartile_3)
    iqr = quartile_3 - quartile_1   # IQR(Inter Quartile Range, 사분범위)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    ic(lower_bound, upper_bound)
    return np.where((data_out > upper_bound) | (data_out < lower_bound))
print(aaa)
outliers_loc = outliers(aaa)
# for i in aaa:
#     print(i)
print('이상치의 위치: ', outliers_loc)

# Visualization
# boxplot
import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()