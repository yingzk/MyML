#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logistic

def classifyVector(inX, weights):
    """
    分类
    """
    prob = logistic.sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    """
    训练和测试模型
    """
    frTrain = open('horse_colic_train.txt')
    frTest = open('horse_colic_test.txt')
    trainingSet = []
    trainingLabels = []
    # --------------训练------------------
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = logistic.randomGradAscent2(np.array(trainingSet), trainingLabels)
    
    # --------------测试------------------
    numTestVec = 0.0
    errorCount = 0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("测试的错误率为: %f" % errorRate)
    return errorRate
      
def multiTest():
    """
    多次测试
    """
    numTests = 10 
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("第 %d 次迭代后，错误率为: %f" % (numTests, errorSum / float(numTests)))
  
if __name__ == '__main__':
    multiTest()