# -*- coding:utf-8 -*-
"""
作者：张少杰
日期：2021年11月11日
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder



#  导入数据集
data = loadmat('../DataSet/ex3data1.mat')
weight = loadmat('../DataSet/ex3weights.mat')
theta1, theta2 = weight['Theta1'],weight['Theta2']


#   sigmoid函数
def sigmoid(z):
    return 1 / (1+np.exp(-z))


#   前向传播函数
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.insert(X,0,values=np.ones(m,),axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2),0,values=np.ones(m),axis=1)
    z3 = a2*theta2.T
    h = sigmoid(z3)

    return a1, z2, a2, z3, h


#   代价函数
def cost(theta1, theta2 , input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    #   执行前向传播
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    #   计算损失
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:],np.log(h[i,:]))
        second_term = np.multiply((1-y[i,:]), np.log(1-h[i,:]))
        J += np.sum(first_term - second_term)

    J = J / m

    return J


# 对y标签进行编码
# 一开始我们得到的y是5000*1维的向量，但我们要把他编码成5000*10的矩阵。
# 比如说，原始y=2，那么转化后的Y对应行就是[0,1,0...0]，原始y=0转化后的Y对应行就是[0,0...0,1]
#
# Scikitlearn有一个内置的编码函数，我们可以使用这个。
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(data['y'])


#   正则化代价函数
def costReg(theta1, theta2, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # 执行前向传播
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # 计算代价函数
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i, :]))
        second_term = np.multiply((1-y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)

    J = J / m

    # 加入正则项
    reg = (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:],2)) + np.sum(np.power(theta2[:,1:],2)))
    J += reg

    return J


# 反向传播
# 这一部分需要实现反向传播的算法，来计算神经网络代价函数的梯度。获得了梯度的数据，我们就可以使用工具库来计算代价函数的最小值。

# sigmoid函数的梯度 g'(x) = g(x) * (1 - g(x)),    其中g(x) = sigmoid(x)
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


#   随机初始化权重参数
# np.random.random(size) 返回size大小的0-1随机浮点数
def init_weight(hidden_size, input_size, num_labels):
    return (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.24


# 进行反向传播
def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # 将权重数组转换成矩阵
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # 进行前向传播
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # 初始化
    J = 0
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)

    # 计算损失函数
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    # 反向传播
    for t in range(m):
        a1t = a1[t,:]  # (1, 401)
        z2t = z2[t,:]  # (1, 25)
        a2t = a2[t,:]  # (1, 26)
        ht = h[t,:]  # (1, 10)
        yt = y[t,:]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1,26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1,26)

        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    return J, delta1, delta2


#   正则化神经网络
def backpropReg(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    # perform backpropagation
    for t in range(m):
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad


# 初始化设置
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1
params = init_weight(hidden_size,input_size,num_labels)
X = data['X']
y = data['y']

#  使用工具库计算参数最优解
fmin = minimize(fun=backpropReg, x0=(params), args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                method='TNC', jac=True, options={'maxiter': 250})

X = np.matrix(X)
thetafinal1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
thetafinal2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

# 计算使用优化后的θ得出的预测
a1, z2, a2, z3, h = forward_propagate(X, thetafinal1, thetafinal2 )
y_pred = np.array(np.argmax(h, axis=1) + 1)
print(classification_report(data['y'], y_pred))