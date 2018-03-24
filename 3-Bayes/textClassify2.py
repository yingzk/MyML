# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def loadDataSet():
    """
    导入数据集
    @return dataset: 数据集
    @return classVec: 分类向量(1代表脏话)
    """
    dataSet = []
    classVec = []
    with open('text.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            dataSet.append(line.split(', ')[:-1])
            classVec.append(line.split(', ')[-1].strip())
    return dataSet, classVec

def createVocabList(dataset):
    """
    创建词汇表
    @param dataset: 数据集
    """
    # 创建集合
    vocabSet = set([])
    for document in dataset:
        # 求并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    """
    @param vocabList: 词表
    @param inputSet: 
    """
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        print(word)

if __name__ == '__main__':
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
