import numpy as np
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X,y=make_classification(n_samples=100,n_features=4,n_redundant=0,random_state=42)
X0=X[y==0]
X1=X[y==1]

#求均值向量
mu0=np.mean(X0,axis=0)
mu1=np.mean(X1,axis=0)

#求类内散度
S0=(X0-mu0).T@(X0-mu0)
S1=(X1-mu1).T@(X1-mu1)
Sw=S0+S1
w_numpy=np.linalg.inv(Sw)@(mu1-mu0)

model=LinearDiscriminantAnalysis()
model.fit(X,y)
w_sklearn=model.coef_[0]

#对最终结果做归一化处理成单位向量
#因为我们最终只关注w方向
print("numpy结果\n",w_numpy/np.linalg.norm(w_numpy))
print("sklear结果\n",w_sklearn/np.linalg.norm(w_sklearn))

