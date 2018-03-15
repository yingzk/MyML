#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def loadDataSet():
    """
    导入数据
    @ return trainData: 训练集
    @ return testData: 测试集
    """
    dataSet = pd.read_csv('watermelon3.0.csv', delimiter=',')
    trainData = dataSet.iloc[:17,:].values
    testData = dataSet.iloc[18, :].values
    return trainData, testData

def resultRatio(dataSet):
    """
    计算各类别所占比
    @return ratios: 各类别比例
    """
    # 结果的类别
    classResult = set(dataSet[:, -1])
    # 各类别的比例
    ratios = {}
    for i in classResult:
        ratios[i] = np.sum(dataSet[:,-1] == i) / len(dataSet)
    return ratios

def splitDataSet(dataSet):
    """
    划分数据集
    @ param dataSet: 数据集
    """
    # 结果的类别
    classResult = set(dataSet[:, -1])
    subDataSet = {}
    # 划分数据集，保存到字典
    for i in classResult:
        subDataSet[i] = [x for x in dataSet if x[-1]==i]
    return subDataSet

def calcFeatureProb(dataSet, testData, feature):
    """
    计算某个类别的概率  (离散)
    @ param dataSet: 训练集
    @ param testData: 测试集
    @ param feature: 属性
    @ return prob: 概率
    """
    featureNums = len([x for x in dataSet if x[feature]==testData[feature]])
    prob = featureNums / len(dataSet)
    print ('属性:', testData[feature], '结果', dataSet[0][-1], 'p',prob)
    return prob

def calcFeatureProb2(dataSet, testData, feature):
    """
    计算某个类别的概率  (离散)
    @ param dataSet: 训练集
    @ param testData: 测试集
    @ param feature: 属性
    @ return prob: 概率
    """
    print (np.var(trainData[:, feature]))
    return (np.exp(-(testData[feature] - np.mean(trainData[:, feature])**2) / 2*np.var(trainData[:, feature])**2) / (np.sqrt(2*np.pi)*np.var(trainData[:, feature])))

if __name__ == '__main__':
    trainData, testData = loadDataSet()
    ratios = resultRatio(trainData)
    subDataSet = splitDataSet(trainData)
    for i in subDataSet.keys():
        s = 1
        for j in range(len(testData)):
            pass