import numpy as np  # noqa: I001
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

#手动放缩
X,y=make_classification(n_samples=2000,n_features=5,weights=[0.95,0.05],random_state=42)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

model_raw=LogisticRegression()
model_raw.fit(X_train,y_train)
y_pred=model_raw.predict(X_test)
print(f"原始模型召回率{recall_score(y_pred=y_pred,y_true=y_test)}")

m_pos=np.sum(y_train==1)
m_neg=np.sum(y_train==0)
y_porb=np.clip(model_raw.predict_proba(X_test)[:, 1], 1e-15, 1 - 1e-15)

rescaled_odds=y_porb/(1-y_porb)*m_neg/m_pos
rescaled_y_prob=(rescaled_odds>1).astype(int)
print(f"模型召回率{recall_score(y_true=y_test,y_pred=rescaled_y_prob)}")

model_sk=LogisticRegression(class_weight='balanced').fit(X_train,y_train)
print(f"模型召回率{recall_score(y_true=y_test,y_pred=model_sk.predict(X_test))}")
