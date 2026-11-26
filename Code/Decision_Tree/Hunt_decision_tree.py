#用纯python实现Hunt算法体系的决策树
from collections import Counter


#1. 构建一个节点TreeNode
class TreeNode:
    """
    label:如果是叶子节点,记录叶子节点的标签
    feature:如果不是叶子节点,记录该节点的属性
    """
    def __init__(self,is_leaf:bool=False,label=None,feature=None) -> None:
        self.is_leaf=is_leaf
        self.label=label
        self.feature=feature
        self.children={}

def all_same_feature(X:list,A:set)->bool:
    if not A or len(X)<=1:
        return True
    first_feature_group=tuple(X[0][a] for a in A)
    for sample in X[1:]:
        feature_group=tuple(sample[a] for a in A)
        if first_feature_group!=feature_group:
            return False
    return True

def count_most_label(y:list)->str|int:
    label=Counter(y).most_common(1)[0][0]
    return label

"""
这里添加属性选择函数(ID3,C4.5,CART)
"""
def find_best_attribute(X:list,A:set):
    return None

#2. 主函数
def generate_tree(data_set:list,features_set:set,feature_value:dict)->TreeNode:
    """
        param:data_set当前节点原始数据集[({属性A:特征,属性B:特征},"标签")......]
        param:features_set:当前可用属性集合
        param:feature_value:{属性:属性值列表}
    """
    node=TreeNode()
    X=[X for X,_ in data_set]
    y=[label for _,label in data_set]

    #1. 第一种递归返回的情况：节点所有样本属于同一个类别
    if len(set(y))==1:
        node.is_leaf=True
        node.label=y[0]
        return node
    #2. 第二种情况：属性集为空或者样本标签不同但是属性值一样
    if not features_set or all_same_feature(X,features_set):
        node.is_leaf=True
        node.label=count_most_label(y)
        return node

    #进入属性选择
    best_attribute=find_best_attribute(X,features_set)
    node.feature=best_attribute
    for value in feature_value[best_attribute]:
        data_value=[(X,y) for X,y in data_set if X[best_attribute] is value]
        if not data_value:
            node.children[value]=TreeNode(is_leaf=True,label=count_most_label(y))
        else:
            node.children[value]=generate_tree(data_value,features_set-{best_attribute},feature_value)
    return node
