# -*- coding:utf-8 -*-
"""
作者：张少杰
日期：2021年10月31日
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas


def sigmod(x, theta, b):
    y = np.sum(x*theta)+b
    if y>0:
        return 1
    else:
        return -1


#   检查是否存在误分点
def check(x, y, theta, b):
    for i in range(y.shape[0]):
        if y[i]*sigmod(x[i], theta, b) <= 0:
            return i
    return None


#   随机梯度下降法
def random_gradient(x, y, theta, b, lr):
    '''

    :param x:
    :param y:
    :param theta: 权重参数
    :param b: 偏执
    :param lr: 学习率
    :return:
    '''
    i = check(x, y, theta, b)
    while i is not None:
        theta = theta + lr*y[i]*x[i]
        b = b + lr*y[i]
        i = check(x, y, theta, b)
    print('stop training:')
    print('权重参数：{}'.format(theta))
    print('偏置：{}'.format(b))
    return theta, b


if __name__ == '__main__':

    #   读取数据
    train_data1 = [[5, 6, 1], [7, 9, 1], [7, 8, 1], [10, 6, 1]]
    train_data2 = [[3, 1, -1], [2, 1, -1], [4, 4, -1], [1, 3, -1], [20, 3, -1]]
    train_datas = train_data1 + train_data2
    train_data1 = np.array(train_data1)
    train_data2 = np.array(train_data2)
    train_datas = np.array(train_datas)

    #   x, y赋值
    x = train_datas[:, [0, 1]]
    y = train_datas[:, [2]]

    #   初始化参数
    theta = np.zeros(2)
    lr = 1
    b = 0

    # 开始训练
    theta, b = random_gradient(x, y, theta, b, lr)
    plt.scatter(train_data2[:, 0], train_data2[:, 1], color='blue', label='-1')
    plt.scatter(train_data1[:, 0], train_data1[:, 1], color='orange', label='1')
    plt.plot(np.arange(20), -(theta[0]*np.arange(20)+b)/theta[1], label='wx+b=0')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()
