from sklearn.datasets import make_classification  # noqa: I001
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier


X,y=make_classification(n_samples=200,n_features=4,n_classes=4,n_informative=4,n_redundant=0,random_state=42)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

base_model=LogisticRegression()
OvO=OneVsOneClassifier(base_model).fit(X_train,y_train)
print(f"OvO训练了{len(OvO.estimators_)}得分{OvO.score(X_train,y_train):.2f}")
OvR=OneVsRestClassifier(base_model).fit(X_train,y_train)
print(f"OvR训练了{len(OvR.estimators_)}得分{OvR.score(X_train,y_train):.2f}")
MvM=OutputCodeClassifier(base_model,code_size=2).fit(X_train,y_train)
print(f"MvM训练了{len(MvM.estimators_)}得分{MvM.score(X_train,y_train):.2f}")
print(MvM.code_book_)
