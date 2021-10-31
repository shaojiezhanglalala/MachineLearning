# -*- coding:utf-8 -*-
"""
作者：张少杰
日期：2021年10月30日
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv
import matplotlib

# 设置中文字体，解决matplotlib无法显示中文字体的问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


#   构建logistic regression, sigmod函数
def sigmod(x):
    return 1./(1+np.exp(-x))


#   梯度下降法
def grdient(X, Y, alpha, theta_g, maxCycles):
    '''

    :param X:
    :param Y:
    :param alpha:学习率
    :param theta_g: 参数向量
    :param maxCycles: 迭代次数
    :return:
    '''
    m = X.shape[0]
    J = pd.Series(np.arange(maxCycles, dtype=float))

    #   开始迭代
    for i in range(maxCycles):
        h = sigmod(np.dot(X, theta_g))
        #   计算损失函数
        J[i] = -(1./m)*np.sum(Y*np.log(h)+(1-Y)*np.log(1-h))
        error = h - Y
        grad = np.dot(X.T, error)
        theta_g -= alpha*grad
    return theta_g, J


#   牛顿法
def Newton(X, Y, theta, maxCycles):
    J = pd.Series(np.arange(maxCycles, dtype=float))
    for i in range(maxCycles):
        h = sigmod(np.dot(X, theta))
        J[i] = -(1./Y.shape[0])*np.sum(Y*np.log(h)+(1-Y)*np.log(1-h))
        error = h - Y
        grad = np.dot(X.T, error)
        A = h*(1-h)*np.eye(len(X))
        H = np.mat(X.T)*A*np.mat(X)
        theta -= inv(H)*grad
    return theta, J


def classifyVector(x, theta):
    h = sigmod(np.dot(X, theta_g))
    return np.where(h >=0.5, 1, 0)


if __name__ == '__main__':
    #   读取文件, 处理数据，将类别信息转换为哑变量
    file_path = '../DataSet/iris.csv'
    iris = pd.read_csv(file_path)
    dummy = pd.get_dummies(iris['Species'])
    iris = pd.concat([iris, dummy], axis=1)

    temp = pd.DataFrame(iris.iloc[:, [1, 2, 3]])
    temp['x0'] = 1
    X = temp.iloc[:, [3, 2, 1, 0]]
    Y = iris[['setosa']]

    # 初始化参数
    alpha = 0.0005
    theta_g = np.zeros((X.shape[1], 1))
    maxCycles = 1000
    theta_g, J = grdient(X, Y, alpha, theta_g, maxCycles)
    print('批量梯度下降法：')
    print('参数向量：{}'.format(theta_g))

    #   绘画迭代次数损失函数J的关系图
    plt.plot(range(maxCycles), J)
    plt.xlabel('迭代次数')
    plt.ylabel('损失函数J')
    plt.show()

    y_p = classifyVector(X, theta_g)
    count = 0
    for i in range(y_p.shape[0]):
        if y_p[i] == Y['setosa'][i]:
            count +=1
    print('预测正确个数：{}'.format(count))
    print('样本个数：{}'.format(Y.shape[0]))
    print('准确率：{0}'.format(count/float(Y.shape[0])))

    # 初始化参数
    alpha = 0.01
    theta_g = np.zeros((X.shape[1], 1))
    theta_g, J = Newton(X, Y, theta_g, maxCycles)
    print('-'*100)
    print('牛顿法')
    print('参数向量：{}'.format(theta_g))
    y_p = classifyVector(X, theta_g)
    count = 0
    for i in range(y_p.shape[0]):
        if y_p[i] == Y['setosa'][i]:
            count +=1
    print('预测正确个数：{}'.format(count))
    print('样本个数：{}'.format(Y.shape[0]))
    print('准确率：{0}'.format(count/float(Y.shape[0])))