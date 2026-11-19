import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

#对于有序的离散属性进行连续化
size = np.array([["S"], ["M"], ["L"], ["XL"]])
ord_enc = OrdinalEncoder(categories=[["S", "M", "L", "XL"]])
print("离散属性连续化结果：\n", ord_enc.fit_transform(size))


#对于无序属性构建n维向量
colors = np.array([["乌黑"], ["青绿"], ["浅白"]])
one_hot = OneHotEncoder(sparse_output=False)
print("离散属性连续化结果：\n", one_hot.fit_transform(colors))
