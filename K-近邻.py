# -*- coding:utf-8 -*-
"""
作者：张少杰
日期：2021年11月02日
"""
import operator

import numpy as np
import matplotlib.pyplot as plt  # 导入相应的数据可视化模块


#   计算x点与X样本矩阵中的点的距离
def distance(x, X):
    return np.sqrt(np.sum((x-X)**2, axis=1))


#   找出前k个与x点最近的点并找出点最多的类别
def find_k_points(x, X, k, label_y):
    X_distances = np.array(distance(x, X))
    print('X_distances:{}'.format(X_distances))
    sortedIndicies = np.argsort(X_distances)[:k]  # 将距离样本按照升序排列,sortedIndicies中存储的是对应前k个元素在X_distances中的索引值
    print('sortedIndicies:{}'.format(sortedIndicies))
    count_dict = {}
    for i in sortedIndicies:
        count_dict[label_y[i]] = count_dict.get(label_y[i], 0)+1
    print('count_dict:{}'.format(count_dict))
    sortedClassCount = sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)
    print("sortedClassCount:{}".format(sortedClassCount))
    return sortedClassCount[0][0]


raw_data_X=[[3.393533211, 2.331273381],
            [3.110073483, 2.781539638],
            [2.343808831, 2.168360954],
            [3.582294042, 2.679179110],
            [2.280362439, 2.866990263],
            [5.423436942, 4.296522875],
            [5.745051997, 3.533989803],
            [6.172168622, 3.711101045],
            [5.792783481, 3.924088941],
            [5.939820817, 4.791637231],
            [3.393533211, 5.331273381],
            [3.110073483, 5.781539638],
            [2.343808831, 5.168360954],
            [3.582294042, 6.679179110],
            [2.280362439, 6.866990263],]
raw_data_Y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
print(raw_data_X)
print(raw_data_Y)
x_train = np.array(raw_data_X)
y_train = np.array(raw_data_Y)  # 数据的预处理，需要将其先转换为矩阵，并且作为训练数据集
print(x_train)
print(y_train)
plt.figure(1)
plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], color="g", label='0')
plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color="r", label='1')   # 将其散点图输出
plt.scatter(x_train[y_train == 2, 0], x_train[y_train == 2, 1], color="orange", label='2')
x = np.array([5.893607318, 4.365731514])  # 定义一个新的点，需要判断它到底属于哪一类数据类型
print('test_point is belong to:{}'.format(find_k_points(x, x_train, 8, y_train)))
plt.scatter(x[0], x[1], color="b", label='test_point')   # 在算点图上输出这个散点，看它在整体散点图的分布情况
plt.axis([0, 8, 0, 8])
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='upper left')
plt.show()