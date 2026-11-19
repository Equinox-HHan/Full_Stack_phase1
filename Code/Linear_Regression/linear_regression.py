import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

"""
模拟:y=wx+b的线性回归:目标是最小化损失函数(最小化残差平方和)
对于LinearRegression:API中的所有参数保持默认即可
训练后得到的参数：
    权重:coef_
    截距:intercept_
    训练时传入的特征数量:n_feature_in_
    可能存在的特征名称:feature_names_in
    矩阵的秩和奇异值:rank_/singular_
常用的函数：
    fit():对线性模型进行拟合
    predict():使用训练好的模型对新数据进行预测
默认不采用梯度下降直接求闭式解
"""
np.random.seed(42)
X=np.random.rand(100,2)*10
y=3*X[:,0]+5*X[:,1]+2+np.random.randn(100)*0.5
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

module=LinearRegression()
module.fit(X_train,y_train,sample_weight=None)


print("权重系数：",module.coef_,)
print("截距：",module.intercept_)

y_pred=module.predict(X_test)
print(y_pred)


"""
对于多特征的线性回归：矩阵求导->令导数为0(涉及到等式左右化简:判断矩阵是否可逆)->求解闭式解
如果我们期望的被同除消掉的矩阵是满秩或者正定矩阵则可逆:可以直接求出权重w
反之如果特征数大于样本数或者特征有多重共线性则需要正则化(用岭回归求解)
"""
np.random.seed(42)
x1=np.random.rand(100,1)*10
x2=2*x1+np.random.randn(100,1)*0.05
X=np.hstack([x1,x2])
y=3*x1+np.random.randn(100,1)*0.01

ridge_module=Ridge(alpha=1)
ridge_module.fit(X,y,sample_weight=None)

print(f"模型的权重和截距:{ridge_module.coef_}-{ridge_module.intercept_}")


#此外还需要注意脏数据污染:这个时候使用Huber回归

np.random.seed(42)
X=np.linspace(0,10,50).reshape(-1,1)
y=2*X.ravel()+1+np.random.randn(50)*0.1
y[-3:]=[1000,2000,3000]

huber_module=HuberRegressor()
huber_module.fit(X,y,sample_weight=None)

print(f"模型的权重和截距:{huber_module.coef_}-{huber_module.intercept_}")

"""
可以用线性回归拟合曲线:本质是人为构建好曲线的形状模型，让计算机尽可能最小化误差
执行过程：
    先执行特征工程将数据升维
    为了防止过拟合和欠拟合要标准化和正则化
"""

np.random.seed(0)
X=np.linspace(-5,5,100).reshape(-1,1)
y=0.5*(X.ravel()**2)+2*X.ravel()+1+np.random.randn(100)*0.1

pipeline=make_pipeline(PolynomialFeatures(degree=2,include_bias=False),StandardScaler(),Ridge(alpha=1))
pipeline.fit(X,y)

print("查看得分：\t",pipeline.score(X,y))
module_pipe=pipeline.named_steps['ridge']
print(module_pipe.coef_,module_pipe.intercept_)
