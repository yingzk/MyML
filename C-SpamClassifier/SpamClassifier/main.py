#!usr/bin/env python
#-*- coding:utf-8 -*-

"""
@author:yzk13
@time: 2018/03/25
"""
import re
from classifier import *

if __name__ == '__main__':
    classifier = Classifier()

    # 定义变量
    # 词频字典
    normalDict = {}
    spamDict = {}
    testDict = {}

    # 保存每封邮件中出现的词
    wordsList = []
    wordsDict = {}

    # 保存结果
    testResult = {}

    # 获取文件夹文件文件名列表
    listNormal = classifier.getFileList(r'data\normal')
    listSpam = classifier.getFileList(r'data\spam')
    # 测试集中文件名低于1000的为正常邮件
    listTest = classifier.getFileList(r'data\test')

    # 正常邮件与垃圾邮件的数量
    numNormal = len(listNormal)
    numSpam = len(listSpam)

    # 获得停词表
    stopList = classifier.getStopWords()

    # 获取正常邮件中的词频
    for fileName in listNormal:
        # 清空词表
        wordsList.clear()
        with open('data/normal/'+fileName, 'r') as f:
            # 过滤掉非中文字符
            for line in f.readlines():
                pattern = re.compile('[^\u4e00-\u9fa5]')
                line = pattern.sub("", line)
                # 将邮件中出现的词保存到wordsList中
                classifier.getWordsList(line, wordsList, stopList)
            # 统计每个词在所有邮件中出现的次数
            classifier.listToDict(wordsList, wordsDict)
    normalDict = wordsDict.copy()

    wordsDict.clear()
    # 获取垃圾邮件中的词频
    for fileName in listSpam:
        # 清空词表
        wordsList.clear()
        with open('data/spam/'+fileName, 'r') as f:
            # 过滤掉非中文字符
            for line in f.readlines():
                pattern = re.compile('[^\u4e00-\u9fa5]')
                line = pattern.sub("", line)
                # 将邮件中出现的词保存到wordsList中
                classifier.getWordsList(line, wordsList, stopList)
            # 统计每个词在所有邮件中出现的次数
            classifier.listToDict(wordsList, wordsDict)
    spamDict = wordsDict.copy()

    # 将结果写入文件wordsProb中,判断是否存在，存在则删除，因为后面执行追加写入
    if os.path.exists('wordsProb.txt'):
        os.remove('wordsProb.txt')

    # 测试邮件
    for fileName in listTest:
        testDict.clear()
        wordsDict.clear()
        wordsList.clear()
        with open('data/test/' + fileName, 'r') as f:
            # 过滤掉非中文字符
            for line in f.readlines():
                pattern = re.compile('[^\u4e00-\u9fa5]')
                line = pattern.sub("", line)
                # 将邮件中出现的词保存到wordsList中
                classifier.getWordsList(line, wordsList, stopList)
            # 统计每个词在所有邮件中出现的次数
            classifier.listToDict(wordsList, wordsDict)
            testDict = wordsDict.copy()

        # 计算每个文件中对分类影响最大的15个词
        wordProbList = classifier.getProbWord(testDict, normalDict, spamDict, numNormal, numSpam)

        # 对每封邮件得到的15个影响率最大的词计算贝叶斯概率
        p = classifier.calBayes(wordProbList, spamDict, normalDict)
        if (p > 0.5):
            testResult[fileName] = 1
        else:
            testResult[fileName] = 0
    testAccuracy = classifier.calAccuracy(testResult)
    # 测试邮件分类结果
    with open('result.txt', 'w') as f:
        for i, ic in testResult.items():
            f.write(str(i) + "," + str(ic) + "\n")

    print('准确率：' + str(testAccuracy))