import numpy as np  # noqa: I001
from collections import Counter
"""
    假设数据分布和西瓜书表格4.3一致
    以此二分类为基础设计三种属性选择函数
"""


def calculate_Ent(data_set:list)->float:
    n=len(data_set)
    if not n:
        return 0.0

    Ent=0.0
    types=Counter(sample[1] for sample in data_set)
    for type in types.values():
        p=type/len(data_set)
        if p>0:
            Ent+=p*np.log2(p)
    
    return -Ent

def calculate_gini(data_set:list)->float:
        n=len(data_set)
        if not n:
            return 0.0

        gini=1.0
        types=Counter(sample[1] for sample in data_set)
        for type in types.values():
            p=type/len(data_set)
            if p>0:
                gini-=p**2

        return gini


def gain(data_set:list,feature_set:set|tuple,feature_values:dict):
    n=len(data_set)
    Ent=calculate_Ent(data_set)
    best_attribute=None
    max_gain=0.0

    for attribute in feature_set:
        Ent_a=0.0
        for value in feature_values[attribute]:
            data_value=[sample for sample in data_set if sample[0][attribute] is value]
            Ent_value=calculate_Ent(data_value)
            Ent_a+=len(data_value)/n*Ent_value

        if max_gain<Ent-Ent_a:
            max_gain=Ent-Ent_a
            best_attribute=attribute

    return best_attribute


def gain_ratio(data_set:list,feature_set:set|tuple,feature_values:dict):
    Ent=calculate_Ent(data_set)
    record=[]

    for attribute in feature_set:
        Ent_a=0.0
        iv_a=0.0
        for value in feature_values[attribute]:
            data_value=[sample for sample in data_set if sample[0][attribute] is value]
            if len(data_value)>0:
                Ent_value=calculate_Ent(data_value)
                weight=len(data_value)/len(data_set)
                Ent_a+=weight*Ent_value
                iv_a-=weight*np.log2(weight)

        gain_a=Ent-Ent_a
        gain_ratio_a=gain_a/iv_a if iv_a>0 else 0.0
        record.append((attribute,gain_a,gain_ratio_a))

    average_gain=np.mean([sample[1] for sample in record])
    gain_list=[sample for sample in record if sample[1]>average_gain]
    gain_list.sort(key=lambda x:x[2],reverse=True)
    return gain_list[0][0]


def gini_index(data_set:list,feature_set:set|tuple,feature_values:dict):
    best_attribute=None
    n=len(data_set)
    min_gini=float("inf")

    for attribute in feature_set:
        gini_a=0.0
        for value in feature_values[attribute]:
            data_value=[sample for sample in data_set if sample[0][attribute] is value]
            if len(data_value)>0:
                weight=len(data_value)/n
                gini_v=calculate_gini(data_value)
                gini_a+=weight*gini_v

        if gini_a<min_gini:
            min_gini=gini_a
            best_attribute=attribute

    return best_attribute
            