#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
实现logistic回归分类算法， 数据集为: dataset.csv
"""

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
   """
   加载数据集
   return: 数据列表， 标签列表
   """
   dataMat = []
   labelMat = []
   # 打开数据集
   fr = open('dataset.csv')
   # 遍历每一行
   for line in fr.readlines():
        # 删除空白符之后进行切分
        lineArr = line.strip().split(',')
        # 数据加入数据列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 标签加入数据列表
        labelMat.append(int(lineArr[2]))
        # 返回数据列表和标签列表
   return dataMat, labelMat
    
def sigmoid(inX):
    """
    计算sigmoid函数
    @: param intX: 矩阵计算的结果(100x1)
    @: return: 计算结果
    """
    return 1.0 / (1 + np.exp(-inX))
        
def gradAscent(dataMat, labelMat):
    """
    梯度上升函数
    @: param dataMat: 数据集
    @: param labelMat: 标签集
    @： return: 权重参数矩阵(最佳回归系数)
    """
    # 将数据转为numpy的数组
    dataMatrix = np.mat(dataMat)
    labelMat = np.mat(labelMat).transpose()
    # 获取矩阵的行列数
    m, n = np.shape(dataMatrix)
    # 初始化参数
    alpha = 0.001
    # 初始化迭代次数
    maxCyc = 500
    # 初始化矩阵的权重参数矩阵， 均为1
    weights = np.ones((n, 1))
    # 开始迭代计算
    for k in range(maxCyc):
        h = sigmoid(dataMatrix * weights)
        # 计算误差
        error = labelMat-h
        # 更新迭代参数
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def plotBestFit(weights):
    """
    绘图
    @:param weights: 系数矩阵
    """
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    # 获取数组行数
    n = np.shape(dataArr)[0]
    # 初始化坐标
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    # 遍历每一行数据
    for i in range(n):
        # 如果对应的类别标签对应数值1，就添加到xcord1，ycord1中
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        # 如果对应的类别标签对应数值0，就添加到xcord2，ycord2中
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    # 添加subplot，三种数据都画在一张图上
    ax = fig.add_subplot(111)
    # 1类用红色标识，marker='s'形状为正方形
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    # 0类用绿色标识，弄认marker='o'为圆形
    ax.scatter(xcord2, ycord2, s=30, c='green')
    # 设置x取值，arange支持浮点型
    x = np.arange(-3.0, 3.0, 0.1)
    # 配计算y的值
    y = (-weights[0]-weights[1]*x)/weights[2]
    # 画拟合直线
    ax.plot(x, y)
    # 贴坐标表头
    plt.xlabel('X1'); plt.ylabel('X2')
    # 显示结果
    plt.show()

def randomGradAscent(dataMat, labelMat):
    """
    随机梯度上升函数
    @: param dataMat: 数据集
    @: param labelMat: 标签集
    @： return: 权重参数矩阵(最佳回归系数)
    """
    dataMatrix = np.array(dataMat)
    m, n = np.shape(dataMatrix)
    # 设置步长
    alpha = 0.01
    # 初始化参数
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        # 计算误差
        error = labelMat[i]-h
        # 更新权重矩阵
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def randomGradAscent2(dataMat, labelMat):
    """
    改进的随机梯度上升算法
    """
    dataMatrix = np.array(dataMat)
    m, n = np.shape(dataMatrix)
    # 初始化参数
    weights = np.ones(n)
    # 迭代次数
    numIter = 500
    for i in range(numIter):
        # 初始化index列表，这里要注意将range输出转换成list
        dataIndex = list(range(m))
        # 遍历每一行数据，这里要注意将range输出转换成list
        for j in list(range(m)):
            # 更新alpha值，缓解数据高频波动
            alpha = 4/(1.0+i+j)+0.0001
            # 随机生成序列号，从而减少随机性的波动
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            # 序列号对应的元素与权重矩阵相乘，求和后再求sigmoid
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            # 求误差，和之前一样的操作
            error = labelMat[randIndex] - h
            # 更新权重矩阵
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 删除这次计算的数据
            del(dataIndex[randIndex])
    return weights

#if __name__ == '__main__':
#    dataArr, labelMat = loadDataSet()
#    # -------------梯度上升---------------
#    weights = gradAscent(dataArr, labelMat)
#    print (weights)
#    plotBestFit(weights.getA())
#    # -------------随机梯度上升-----------
#    weights2 = np.mat(randomGradAscent(dataArr, labelMat)).transpose()
#    print (weights2)
#    plotBestFit(weights2.getA())
#    # -------------改进随机梯度上升-----------
#    weights3 = np.mat(randomGradAscent2(dataArr, labelMat)).transpose()
#    print (weights3)
#    plotBestFit(weights3.getA())