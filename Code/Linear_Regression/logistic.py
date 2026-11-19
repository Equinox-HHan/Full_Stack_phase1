import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X,y=make_classification(n_samples=200,n_features=2,n_redundant=0,random_state=42)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

def sigmoid(z:np.ndarray)->np.ndarray:
    result=1/(1+np.exp(-z))
    return result

m,n_feature=X_train.shape
w=np.zeros(n_feature)
b=0.00
learning_rate=0.1
epoch=1000

for i in range(epoch):
    # 前向计算概率
    z=X_train@w+b
    p=sigmoid(z)

    # 计算交叉熵
    loss=-np.mean(y_train*np.log(p+1e-15)+(1-y_train)*np.log((1-p)+1e-15))

    # 反向计算梯度
    error=p-y_train
    dirta_w=(X_train.T@error)/m
    dirta_b=np.mean(error)

    w=w-learning_rate*dirta_w
    b=b-learning_rate*dirta_b

    if i%100==0:
        print(f"当前参数{w}-{b}")

prob=sigmoid(X_test@w+b)
y_pred=(prob>=0.5).astype(int)
acc=np.mean(y_pred==y_test)

print("numpy方式求解")
print("学到的 w:", w, "b:", b)
print(f"测试集准确率: {acc:.2%}")


model=LogisticRegression()
model.fit(X_train,y_train)

y_pred_sk = model.predict(X_test)              
y_prob_sk = model.predict_proba(X_test)        
acc_sk = model.score(X_test, y_test)           

print("\n[Sklearn 版结果]")
print("学到的 w:", model.coef_[0], "b:", model.intercept_[0])
print(f"测试集准确率: {acc_sk:.2%}")