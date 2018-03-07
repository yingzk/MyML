#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

#-------------欧式距离`------------
def euclidean2(a, b):
    """
    二维空间
    """
    distance = np.sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2 )
    return distance
print ('二维空间a, b两点之间的欧式距离为： ', euclidean2((1,1),(2,2)))

def euclidean3(a, b):
    """
    三维空间
    """
    distance = np.sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2 )
    return distance
print ('三维空间a, b两点之间的欧式距离为： ', euclidean3((1,1,1),(2,2,2)))

def euclideann(a, b):
    """
    n维空间
    """
    sum = 0
    for i in range(len(a)):
        sum += (a[i]-b[i])**2
    distance = np.sqrt(sum)
    return distance
print ('n 维空间a, b两点之间的欧式距离为： ', euclideann((1,1,2,2),(2,2,4,4)))

def euclideann2(a, b):
    """
    n维空间, 不使用循环
    """
    A = np.array(a)
    B = np.array(b)
    c = (A - B) **2
    distance = np.sqrt(sum(c))
    return distance
print ('n 维空间a, b两点之间的欧式距离为： ', euclideann2((1,1,2,2),(2,2,4,4)))


def euclideans(a, b):
    """
    标准化欧氏距离
    """
    sumnum = 0
    for i in range(len(a)):
        # 计算si 分量标准差
        avg = (a[i] - b[i]) / 2
        si = np.sqrt( (a[i]-avg)**2 + (b[i]-avg)**2 )
        sumnum += ((a[i]-b[i]) / si) **2
    distance = np.sqrt(sumnum)
    return distance
print ('a, b两点的标准化欧氏距离为： ', euclideann2((1,2,1,2),(3,3,3,4)))

#-------------曼哈顿距离`------------

def manhattan2(a, b):
    """
    二维空间曼哈顿距离
    """
    distance = np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])
    return distance
print ('二维空间a, b两点之间的曼哈顿距离为： ', manhattan2((1,1),(2,2)))

def manhattann(a, b):
    """
    n维空间曼哈顿距离
    """
    distance = 0 
    for i in range(len(a)):
        distance += np.abs(a[i]-b[i])
    return distance
print ('n维空间a, b两点之间的曼哈顿距离为： ', manhattann((1,1,2,2),(2,2,4,4)))

def manhattann2(a, b):
    """
    n维空间曼哈顿距离, 不使用循环
    """
    A = np.array(a)
    B = np.array(b)
    distance = sum(np.abs(A-B))
    return distance
print ('n维空间a, b两点之间的曼哈顿距离为： ', manhattann2((1,1,2,2),(2,2,4,4)))

#-------------切比雪夫距离`------------
def chebyshev2(a, b):
    """
    二维空间切比雪夫距离
    """
    distance = max(abs(a[0]-b[0]), abs(a[1]-b[1]))
    return distance
print ('二维空间a, b两点之间的切比雪夫距离为： ', chebyshev2((1,2),(3,4)))

def chebyshevn(a, b):
    """
    n维空间切比雪夫距离
    """
    distance = 0
    for i in range(len(a)):
        if (abs(a[i]-b[i]) > distance):
            distance = abs(a[i]-b[i])
    return distance
print ('n维空间a, b两点之间的切比雪夫距离为：' , chebyshevn((1,1,1,1),(3,4,3,4)))

def chebyshevn2(a, b):
    """
    n维空间切比雪夫距离, 不使用循环
    """
    distance = 0
    A = np.array(a)
    B = np.array(b)
    distance = max(abs(A-B))
    return distance
print ('n维空间a, b两点之间的切比雪夫距离为：' , chebyshevn2((1,1,1,1),(3,4,3,4)))

#-------------闵可夫斯基距离`------------

def minkowski(a, b):
    """
    闵可夫斯基距离
    """
    A = np.array(a)
    B = np.array(b)
    #方法一：根据公式求解
    distance1 = np.sqrt(np.sum(np.square(A-B)))
    
    #方法二：根据scipy库求解
    from scipy.spatial.distance import pdist
    X = np.vstack([A,B])
    distance2 = pdist(X)[0]
    return distance1, distance2
print ('二维空间a, b两点之间的闵可夫斯基距离为：' , minkowski((1,1),(2,2))[0])

#-------------马氏距离`------------
def mahalanobis (a, b):
    """
    马氏距离
    """
    A = np.array(a)
    B = np.array(b)
    #马氏距离要求样本数要大于维数，否则无法求协方差矩阵
    #此处进行转置，表示10个样本，每个样本2维
    X = np.vstack([A,B])
    XT = X.T
    
    #方法一：根据公式求解
    S = np.cov(X)   #两个维度之间协方差矩阵
    SI = np.linalg.inv(S) #协方差矩阵的逆矩阵
    #马氏距离计算两个样本之间的距离，此处共有10个样本，两两组合，共有45个距离。
    n = XT.shape[0]
    distance1 = []
    for i in range(0, n):
        for j in range(i+1, n):
            delta = XT[i] - XT[j]
            d = np.sqrt(np.dot(np.dot(delta,SI),delta.T))
            distance1.append(d)
            
    #方法二：根据scipy库求解
    from scipy.spatial.distance import pdist
    distance2 = pdist(XT,'mahalanobis')
    return  distance1, distance2
print ('(1, 2)，(1, 3)，(2, 2)，(3, 1)两两之间的闵可夫斯基距离为：' , mahalanobis((1, 1, 2, 3),(2, 3, 2, 1))[0])

#-------------夹角余弦`------------
def cos2(a, b):
    """
    二维夹角余弦
    """
    cos = (a[0]*b[0] + a[1]*b[1]) / (np.sqrt(a[0]**2 + a[1]**2) * np.sqrt(b[0]**2+b[1]**2))
    return cos
print ('a,b 二维夹角余弦距离：',cos2((1,1),(2,2)))

def cosn(a, b):
    """
    n维夹角余弦
    """
    sum1 = sum2 = sum3 = 0
    for i in range(len(a)):
        sum1 += a[i] * b[i]
        sum2 += a[i] ** 2
        sum3 += b[i] ** 2
    cos = sum1 / (np.sqrt(sum2) * np.sqrt(sum3))
    return cos
print ('a,b 多维夹角余弦距离：',cosn((1,1,1,1),(2,2,2,2)))

def cosn2(a, b):
    """
    n维夹角余弦, 不使用循环
    """
    A, B = np.array(a), np.array(b)
    sum1 = sum(A * B)
    sum2 = np.sqrt(np.sum(A**2))
    sum3 = np.sqrt(np.sum(B**2))
    cos = sum1 / (sum2 * sum3)
    return cos
print ('a,b 多维夹角余弦距离：',cosn2((1,1,1,1),(2,2,2,2)))

#-------------汉明距离`------------
def hamming(a, b):
    """
    汉明距离
    """
    sumnum = 0
    for i in range(len(a)):
        if a[i]!=b[i]:
            sumnum += 1
    return sumnum
print ('a,b 汉明距离：',hamming((1,1,2,3),(2,2,1,3)))

def hamming2(a, b):
    """
    汉明距离, 不使用循环
    """
    matV = np.array(a) - np.array(b)
    numsum = len(np.nonzero(matV)[0])
    return numsum
print ('a,b 汉明距离：',hamming2((1,1,2,3),(2,2,1,3)))

#------------杰卡德相似系数 & 杰卡德距离`-------------
def jaccard_coefficient(a, b):
    """
    杰卡德相似系数
    """
    set_a = set(a)
    set_b = set(b)
    distance = float(len(set_a & set_b)) / len(set_a | set_b)
    return distance
print ('a,b 杰卡德相似系数：', jaccard_coefficient((1,2,3),(2,3,4)))

def jaccard_distance(a, b):
    """
    杰卡德距离
    """
    set_a = set(a)
    set_b = set(b)
    distance = float(len(set_a | set_b) - len(set_a & set_b)) / len(set_a | set_b)
    return distance
print ('a,b 杰卡德距离：', jaccard_coefficient((1,2,3),(2,3,4)))

#------------相关系数 & 相关距离`-------------
def correlation_coefficient():
    """
    相关系数
    """
    a = np.array([[1, 1, 2, 2, 3], [2, 2, 3, 3, 5], [1, 4, 2, 2, 3]])
    print ('a的行之间相关系数为： ', np.corrcoef(a))
    print ('a的列之间相关系数为： ', np.corrcoef(a,rowvar=0))
correlation_coefficient()

def correlation_distance():
    """
    相关距离
    """
    a = np.array([[1, 1, 2, 2, 3], [2, 2, 3, 3, 5], [1, 4, 2, 2, 3]])
    print ('a的行之间相关距离为： ', np.ones(np.shape(np.corrcoef(a)),int) - np.corrcoef(a))
    print ('a的列之间相关距离为： ', np.ones(np.shape(np.corrcoef(a,rowvar = 0)),int) - np.corrcoef(a,rowvar = 0))
correlation_distance()

#------------信息熵`-------------
def calc_entropy(x):
    """
    计算信息熵
    """
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent

def calc_condition_entropy(x, y):
    """
    计算条件信息熵
    """

    # calc ent(y|x)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_entropy(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent
    return ent

def calc_entropy_grap(x,y):
    """
    计算信息增益
    """

    base_ent = calc_entropy(y)
    condition_ent = calc_condition_entropy(x, y)
    ent_grap = base_ent - condition_ent
    return ent_grap