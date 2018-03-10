#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import operator

def loadDataSet():
    """
    导入数据
    @ return dataSet: 读取的数据集
    """
    # 对数据进行处理
    dataSet = pd.read_csv('isFish.csv', delimiter=',')
    # dataSet = dataSet.replace('yes', 1).replace('no', 0)
    labelSet = list(dataSet.columns.values)
    dataSet = dataSet.values
    return dataSet, labelSet

def calcShannonEnt(dataSet):
    """
    计算给定数据集的信息熵（香农熵）
    @ param dataSet: 数据集
    @ return shannonEnt: 香农熵
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        # 当前样本类型
        currentLabel = featVec[-1]
        # 如果当前类别不在labelCounts里面，则创建
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob*np.log2(prob)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    """
    划分数据集, 提取所有满足一个特征的值
    @ param dataSet: 数据集
    @ param axis: 划分数据集的特征
    @ param value: 提取出来满足某特征的list
    """
    retDataSet = []
    for featVec in dataSet:
        # 将相同数据特征的提取出来
        if featVec[axis] == value:
            reducedFeatVec = list(featVec[:axis])
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeature(dataSet):
    """
    选择最优的划分属性
    @ param dataSet: 数据集
    @ return bestFeature: 最佳划分属性
    """
    # 属性的个数
    numFeature = len(dataSet[0])-1
    baseEntroy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeature):
        # 获取第i个特征所有可能的取值
        featureList = [example[i] for example in dataSet]
        # 去除重复值
        uniqueVals = set(featureList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # 特征为i的数据集占总数的比例
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * np.log2(prob)
        inforGain = baseEntroy - newEntropy
        
        if inforGain > bestInfoGain:
            bestInfoGain = inforGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    """
    递归构建决策树
    @ param classList: 类别列表
    @ return sortedClassCount[0][0]: 出现次数最多的类别
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount += 1
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """
    构造决策树
    @ param dataSet: 数据集
    @ param labels: 标签集
    @ return myTree: 决策树
    """
    classList = [example[-1] for example in dataSet]
    # 当类别与属性完全相同时停止
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征值时，返回数量最多的
    if (len(dataSet[0]) == 1):
        return majorityCnt(classList)
    
    # 获取最佳划分属性
    bestFeat = chooseBestFeature(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    # 清空labels[bestFeat]
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        # 递归调用创建决策树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
    

if __name__ == '__main__':
    dataSet, labelSet = loadDataSet()
    shannonEnt = calcShannonEnt(dataSet)
    tree= createTree(dataSet, labelSet)
    print (tree)
    
    