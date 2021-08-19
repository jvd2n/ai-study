# 이상치 처리
# 1. 삭제
# 2. Nan 처리 후 -> 보간   // linear
# 3. ......... (결측치 처리 방법과 유사)
# 4. scaler -> Robust, QuantileTransformer 등
# 5. 모델링: tree 계열, DT, RF, XG, LGBM ...


import numpy as np
aaa = np.array([1, 2, -1000, 4, 5, 6, 7, 8, 90, 100, 500])
print(aaa)
print(aaa.shape)
print(type(aaa))
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print('1사분위: ', quartile_1)
    print('q2 :', q2)
    print('3사분위: ', quartile_3)
    iqr = quartile_3 - quartile_1   # IQR(Inter Quartile Range, 사분범위)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound) | (data_out < lower_bound))

outliers_loc = outliers(aaa)

print('이상치의 위치: ', outliers_loc)


# Visualization
# boxplot
import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()