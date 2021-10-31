# -*- coding:utf-8 -*-
"""
作者：张少杰
日期：2021年10月27日
"""
import random


import numpy as np


# 生成数组
t1 = np.array([1, 2, 3])
print(t1, type(t1))
t2 = np.array(range(10))
print(t2, type(t2))
t3 = np.arange(4, 10, 2)
print(t3, type(t3), t3.dtype)
t4 = np.array(range(1, 25), dtype='i1')
print(t4, type(t4), t4.dtype)
t5 = np.array([random.random() for i in range(10)])
print(t5, type(t5), t5.dtype)
# 取小数 2为保留小数的位数
t6 = np.round(t5, 2)
print(t6, type(t6), t6.dtype)
# 查找数组形状 shape
print(t1.shape)
t7 = np.array([[1, 2, 3], [4, 5, 6]])
print(t7.shape)
# 改变数组形状 reshape 并不改变元数组的形状，而是产生一个新的数组
print(t2.reshape(2, 5))
print(t4.reshape(2, 3, 4))
print(t2, t2*2)

# 矩阵转置
t8 = np.array([[0, 1, 2, 3, 4],
 [5, 6, 7, 8, 9]])
print(t8.transpose())
print(t8.T)
print(t8.swapaxes(1, 0))
# numpy读数据
file_path = 'dataSet/studentscores.csv'
f1 = np.loadtxt(file_path, dtype=float, delimiter=',',skiprows=1)
print(f1)
# numpy 索引和切片
# 取行
print(f1[1])
# 取多行
print(f1[1:])
# 取不连续的多行
print(f1[[1, 4, 6]])
# 取列
print(f1[:, 1])
# 取多列
print(f1[:, 0:])
# 取不连续多列
# print(f1[:,[1, 3, 5])

# 对矩阵满足条件的元素进行修改
print(t8[t8>5]+1)
# 三元运算符 np.where(条件.满足条件，不满足条件)
# t.clip（10，18）小于10替换为10，大于18的替换为18

# 转换矩阵元素类型
t9 = t8.astype(float)
print(t9)
# 将浮点数元素变成nan
t9[1,1] = np.nan
print(t9)

# 数组拼接
t1 = np.array([[0, 1, 2, 3, 4],
 [5, 6, 7, 8, 9]])
t2 = np.array([[0, 1, 2, 3, 4],
 [5, 6, 7, 8, 9]])+1
print(t1)
print(t2)
t3 = np.vstack((t1, t2))  # 竖直拼接 上下拼接
t4 = np.hstack((t1, t2))  # 水平拼接  左右拼接
print(t3)
print(t4)

# 行列交换
t1[[0, 1], :] = t1[[1, 0], :]
print(t1)
t1[:, [0, 1]] = t1[:, [1, 0]]
print(t1)

# 创建全为0或1的数组
t1 = np.zeros((2, 3), dtype=int)
t2 = np.ones((3, 4), dtype=float)
t3 = np.eye(3)  # 单位矩阵
print(t1)
print(t2)
print(t3)

# 找到数字最大最小值的索引位置
t2 = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
print(t2)
ans1 = np.argmax(t2, axis=0)
ans2 = np.argmin(t2, axis=1)
print(ans1)
print(ans2)

# numpy生成随机数数组
np.random.seed(10)  # 添加随机种子，使每次随机结果相同
t1 = np.random.randint(10, 20, (4, 5))
t2 = np.random.rand(3, 4)  # 均匀分布随机数
t3 = np.random.randn(3, 4)  # 正态分布随机数 0-1 浮点数 标准差为1，μ = 0
t4 = np.random.uniform(10, 20, (3, 4))  # （最小值，最大值，数组形状）
t5 = np.random.normal(2, 1, (3, 4))  # （μ，标准差，数组形状）
print(t1)
print(t2)
print(t3)
print(t4)
print(t5)

# 关于nan和inf
'''
1.两个nan不相等
2.nan！=nan 可以利用该特性判断数组中nan的个数 np.count.nonzero(t!=t)
3.np.isnan()
4.一般会将nan替换为均值或中值
'''

# np.sum(t1,axis=) axis不写则计算数组所有元素的值
t1 = np.arange(20).reshape((4,5))
sum1 = np.sum(t1)
sum2 = np.sum(t1, axis=0)  # 算每一列值的和
sum3 = np.sum(t1, axis=1)  # 算每一行值的和
print(sum1)
print(sum2)
print(sum3)
# np.ptp(t, axis= )求极值
# 标准差 t.std(axis=) t为数组
