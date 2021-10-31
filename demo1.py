# -*- coding:utf-8 -*-
"""
作者：张少杰
日期：2021年10月29日
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# 设置中文字体，解决matplotlib无法显示中文字体的问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# 计算损失函数
def liner_loss(w, b, x, y):
    """

    :param w: 线性函数系数
    :param b: 截距
    :param x: x值
    :param y: y值
    :return:
    """
    # 损失函数：使用的是均方误差（MES）损失
    loss = np.sum((y - w * x - b) ** 2) / len(x)
    return loss


# 梯度下降
def liner_gradient(w, b, x, y, lr):
    """

    :param w: 线性函数系数
    :param b: 截距
    :param x: x值
    :param y: y值 向量
    :param lr: 学习率
    :return: 返回更新后的参数w，b
    """
    n = float(len(x))
    #     求梯度
    dw = np.sum(-(2 / n) * x * (y - w * x - b))
    db = np.sum(-(2 / n) * (y - w * x - b))
    # 更新参数
    w = w - lr * dw
    b = b - lr * db
    return w, b


# 迭代梯度下降
def optimer(x, y, w, b, lr, epcoh):
    """

    :param x: x值 向量
    :param y: y值 向量
    :param w: 参数
    :param b: 截距
    :param lr: 学习率
    :param epcoh: 迭代次数
    :return: 多次迭代梯度下降后的w, b
    """
    for i in range(epcoh):
        w, b = liner_gradient(w, b, x, y, lr)
        #        没累计迭代100次计算一次损失函数
        if i % 100 == 0:
            print('epoch {0}:loss={1}'.format(i, liner_loss(w, b, x, y)))

    return w, b


# 绘图
def plot_data(x, y, w, b):
    y_predict = w * x + b
    plt.plot(x, y, 'o', label='训练数据')
    plt.plot(x, y_predict, 'k', label='预测模型')
    plt.title('学习时长和学习成绩的关系')
    plt.xlabel('学习时长 单位：小时')
    plt.ylabel('学习成绩 单位: 分')
    plt.legend()
    plt.show()


# 构建线性回归模型
if __name__ == '__main__':
    dataset_path = '../DataSet/studentscores.csv'
    data = np.loadtxt(dataset_path, dtype=float, delimiter=',', skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y, 'o')
    plt.show()

    #     初始化参数
    lr = 0.01
    epoch = 1000
    w = 0.0
    b = 0.0
    # 输出各个参数初始值
    print('initial variables:\n initial_b = {0}\n intial_w = {1}\n loss of begin = {2} \n' \
          .format(b, w, liner_loss(w, b, x, y)))

    w, b = optimer(x, y, w, b, lr, epoch)
    # 输出各个参数的最终值
    print('final formula parmaters:\n b = {1}\n w = {2}\n loss of end = {3} \n'.format(epoch, b, w,
                                                                                       liner_loss(w, b, x, y)))
    # 显示
    plot_data(x, y, w, b)