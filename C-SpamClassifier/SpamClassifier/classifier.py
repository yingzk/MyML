#!usr/bin/env python  
# -*- coding:utf-8 -*-

""" 
@author:yzk13 
@time: 2018/03/25 
"""
import jieba
import os

class Classifier:
    """
    分类器
    """
    def getStopWords(self):
        """
        获得停词列表
        :return stopList 停词列表
        """
        stopList = []
        with open('data/stopWords.txt', 'r') as f:
            for line in f.readlines():
                # 去除末尾\n
                stopList.append(line[:-1])
        return stopList

    def getFileList(self, filePath):
        """
        获取文件名称列表
        :param: filePath 文件夹路径
        :return fileNameList 文件名称列表
        """
        fileNameList = os.listdir(filePath)
        return fileNameList

    def getWordsList(self, content, wordsList, stopList):
        """
        获取词表词典, 由于要对多文本进行统计，所以这里不返回，一直在后面添加
        :param content: 文本内容
        :param wordsList: 词表
        :param stopList: 停词表
        """
        resultList = list(jieba.cut(content))
        for result in resultList:
            # 停词
            if result not in stopList and result.strip() != '' and result != None:
                if result not in wordsList:
                    wordsList.append(result)

    def listToDict(self, wordsList, wordsDict):
        """
        list转dict
        :param wordsList: 词列表
        :param wordsDict: 词字典
        """
        for item in wordsList:
            if item in wordsDict.keys():
                wordsDict[item] += 1
            else:
                wordsDict[item] = 1

    def getProbWord(self, testDict, normalDict, spamDict, numNormal, numSpam):
        """
        计算对分类结果影响最大的15个词
        :param testDict: 测试数据字典
        :param normalDict: 正常邮件字典
        :param spamDict: 垃圾邮件字典
        :param numNormal: 正常邮件的数量
        :param numSpam: 垃圾邮件的数量
        :return wordProbList: 对分类结果影响最大的15个词
        """
        wordProbList = {}
        for word, num in testDict.items():
            # 当词不在垃圾邮件词表中，在正常邮件词表中，计算概率
            if word in spamDict.keys() and word in normalDict.keys():
                # 求类先验概率
                # 正常邮件
                pw_n = normalDict[word] / numNormal
                # 垃圾邮件
                pw_s = spamDict[word] / numSpam
                ps_w = pw_s / (pw_s + pw_n)
                wordProbList[word] = ps_w
            # 当词在垃圾邮件词表中，不在正常邮件词表中，计算概率
            if word in spamDict.keys() and word not in normalDict.keys():
                pw_s = spamDict[word] / numSpam
                pw_n = 0.01
                ps_w = pw_s / (pw_s + pw_n)
                wordProbList[word] = ps_w
            # 当词在垃圾邮件词表中，而且在正常邮件词表中，计算概率
            if word not in spamDict.keys() and word in normalDict.keys():
                pw_s = 0.01
                pw_n = normalDict[word] / numNormal
                ps_w = pw_s / (pw_s + pw_n)
                wordProbList[word] = ps_w
            # 当词不在垃圾邮件词表中，也不在正常邮件词表中，计算概率
            if word not in spamDict.keys() and word not in normalDict.keys():
                wordProbList[word] = 0.5  # 0.4
        sorted(wordProbList.items(), key=lambda d: d[1], reverse=True)[0:15]
        return wordProbList

    def calBayes(self, wordList, spamDict, normalDict):
        """
        计算贝叶斯概率
        :param wordList: 词表
        :param spamDict: 垃圾邮件词语字典
        :param normalDict: 正常邮件词语字典
        :return: 概率
        """
        ps_w = 1
        ps_n = 1

        with open('wordsProb.txt', 'a', encoding='utf-8') as f:
            for word, prob in wordList.items():
                f.write(word + ":" + str(prob) + "\n")
                ps_w *= prob
                ps_n *= 1 - prob
            p = ps_w / (ps_w + ps_n)
        return p

    def calAccuracy(self, testResult):
        """
        计算精度
        :return:
        """
        rightCount = 0
        errorCount = 0
        for name, catagory in testResult.items():
            if (int(name) < 1000 and catagory == 0) or (int(name) > 1000 and catagory == 1):
                rightCount += 1
            else:
                errorCount += 1
        return rightCount / (rightCount + errorCount)
