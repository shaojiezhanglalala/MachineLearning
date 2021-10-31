# -*- coding:utf-8 -*-
"""
作者：张少杰
日期：2021年10月29日
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib


# 设置中文字体，解决matplotlib无法显示中文字体的问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# 计算损失函数
def liner_loss(w, b, x, y):
    predict_y = np.dot(x, w)
    delta_y = predict_y + b - y
    loss = np.dot(delta_y.T, delta_y)/x.shape[0]
    return loss


# 梯度下降算法
def multiple_linear_gradient(w, b, x, y, lr):
    """

    :param w: 线性函数系数
    :param b: 截距
    :param x: x值 向量
    :param y: y值 向量
    :param lr: 学习率
    :return: 返回更新后的参数w，b
    """
    n = y.shape[0]
    #   对多个x变量依次求梯度
    dw = (-2) / n * np.dot(x.T, (y - np.dot(x, w) - b))
    db = (-2) / n * np.sum(y - np.dot(x, w) - b)
    #   更新参数
    w = w - lr * dw
    b = b - lr * db
    return w, b


#   正规方程法
def Normal_equation_method(x, y, w):
    w = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
    b = w[0]
    w = w[1:]
    return w, b


#   迭代梯度下降
def optimer(w, b, x, y, lr, epcoh):
    '''

    :param w:
    :param b:
    :param x:
    :param y:
    :param lr:
    :param epcoh:迭代次数
    :return:
    '''
    for i in range(epcoh):
        w, b = multiple_linear_gradient(w, b, x, y, lr)
        # if i % 100 == 0:
        #     print('epoch {0}: loss={1}'.format(i, liner_loss(w, b, x, y)))

    return w, b


#   构建多元线性回归模型
if __name__ == '__main__':
    data_path = '../DataSet/airline_reviews.csv'

    #   读取数据
    df = pd.read_csv(data_path)
    # print(df.head())
    # print(df.info())
    data = pd.DataFrame(df)
    x = np.array(data[['Food_&_Beverages', 'Inflight_Entertainment', 'Seat_Comfort', 'Staff_Service', 'Value_for_Money', 'Review_Count']])
    y = np.array(data[['Rating']])


    # 初始化参数
    lr = 0.000004
    epoch = 1000000
    w = np.zeros((x.shape[1], 1))
    b = 0
    print('initial_loss:{0}'.format(liner_loss(w, b, x, y)))

    # 训练
    #   梯度下降
    w, b = optimer(w, b, x, y, lr, epoch)
    print('gradient_method:')
    print('w:{0}, b:{1}'.format(w, b))
    print('loss:{}'.format(liner_loss(w, b, x, y)))

    #   正规方程
    b = np.ones((x.shape[0], 1))
    x = np.hstack((b, x))
    w = np.zeros((x.shape[1]+1, 1))
    w, b = Normal_equation_method(x, y, w)
    print('-'*100)
    print('normal_equation_method:')
    print('w:{0}, b:{1}'.format(w, b))
    print('loss:{}'.format(liner_loss(w, b[0], x[:, 1:], y)))
