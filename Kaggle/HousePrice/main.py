#!usr/bin/env python  
# -*- coding:utf-8 -*-

""" 
@author:yzk13 
@time: 2018/04/05 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

train_df = pd.read_csv('input/train.csv', index_col=0)
test_df = pd.read_csv('input/test.csv', index_col=0)

price = pd.DataFrame({'price': train_df['SalePrice'], 'log(price+1)' : np.log1p(train_df['SalePrice'])})
# 查看数据是否平滑
# price.hist()
# plt.show()
# 提取出需要预测的列 y_train
y_train = np.log1p(train_df.pop('SalePrice'))

# 合并数据
# print(train_df.shape)
# print(test_df.shape)
all_df = pd.concat((train_df, test_df), axis=0)
print(all_df.shape)

# 特征工程
print(all_df.dtypes)
# 将MSSubClass 属性转换为str型，因为它是类别不是数字，对于这个属性是一个classification问题
all_df['MSSubClass'] = all_df['MSSubClass'].astype('str')
print(all_df['MSSubClass'].dtype)
# 对每个类型进行统计
print(all_df['MSSubClass'].value_counts())

# One-Hot 独热码，将类别转换为独热码
all_dummy_df = pd.get_dummies(all_df)

# 处理缺失值
print(all_dummy_df.isnull().sum().sort_values(ascending=False).head(10))
# 用平均值填补空缺
mean_cols = all_dummy_df.mean()
all_dummy_df = all_dummy_df.fillna(mean_cols)

# 对原来数据中数值型的数据进行平滑处理
numeric_cols = all_df.columns[all_df.dtypes != 'object']
numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std

# 把数据集分回去
dummy_train = all_dummy_df.loc[train_df.index]
dummy_test = all_dummy_df.loc[test_df.index]
print(dummy_train.shape)
print(dummy_test.shape)

# 将数据转换为numpy个数
X_train = dummy_train.values
X_test = dummy_test.values

# 模型训练
def RidgeFunc():
    """
    Ridge
    :return:
    """
    alphas = np.logspace(-3, 2, 100)
    test_scores = {}
    for alpha in alphas:
        clf = Ridge(alpha)
        # 我们希望error越小
        test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
        test_scores[alpha] = np.mean(test_score)
    print(min([x for x in test_scores.values()]))
    print(test_scores)
    plt.plot(alphas, [x for x in test_scores.values()])
    plt.title('Alpha VS Error CV')
    plt.show()

def RandomForestFunc():
    """
    随机森林
    :return:
    """
    # 每棵树最大能看到的属性百分比
    max_features = [.1, .3, .5, .7, .9, .99]
    test_scores = {}
    for feature in max_features:
        clf = RandomForestRegressor(n_estimators=200, max_features=feature)
        test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
        test_scores[feature] = np.mean(test_score)
    print(min([x for x in test_scores.values()]))
    print(test_scores)
    plt.plot(max_features, [x for x in test_scores.values()])
    plt.title('Max Features VS Error CV')
    plt.show()

# 得到最好的alpha为17.4
# RidgeFunc()
# 得到最好的max_features为0.3
# RandomForestFunc()

ridge = Ridge(alpha=17.4)
ridge.fit(X_train, y_train)
y_ridge = np.expm1(ridge.predict(X_test))

rd = RandomForestRegressor(n_estimators=50, max_features=0.3)
rd.fit(X_train, y_train)
y_rd = np.expm1(rd.predict(X_test))

y_final = (y_ridge + y_rd) / 2
submission_df = pd.DataFrame(data={'Id':test_df.index, 'SalePrice':y_final})
submission_df.to_csv('result.csv', index=False)