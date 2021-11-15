# -*- coding:utf-8 -*-
"""
作者：张少杰
日期：2021年11月11日
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat


#   定义PCA函数
def pca(X):
    #   将X中的数据归一化
    X = (X - X.mean())/X.std()
    #   计算X的协方差矩阵
    X = np.matrix(X)
    cov = (X.T * X)/X.shape[0]
    #   对协方差矩阵进行SVD奇异值分解
    U,S,V = np.linalg.svd(cov)
    return U,S,V


#   通过PCA得到的矩阵U为主成分，根据要求选取前k列作为k个主成分，对矩阵X进行压缩
def project_data(X,U,k):
    U_reduce = U[:,:k]
    return np.dot(X,U_reduce)


#   可以通过反向步骤对压缩的原始数据进行恢复
def recover_data(Z,U,k):
    U_reduced = U[:,:k]
    return np.dot(Z,U_reduced.T)


#   导入并处理数据
data = loadmat('../DataSet/ex7data1.mat')
# print(data)

X = data['X']
# print(X)

#   画出X的原始分布图
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:, 0], X[:, 1],color='blue',label='source_data')

#   利用pca函数得到X的协方差矩阵的奇异值分解结果
U,S,V = pca(X)

# 接收压缩后的特征矩阵
Z = project_data(X,U,1)

#   根据压缩数据Z，使用recover_data恢复原始数据
X_recovered = recover_data(Z,U,1)

#   画出经压缩后恢复的数据X
ax.scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]),color='green',label='recovered_data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()




