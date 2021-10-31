# -*- coding:utf-8 -*-
"""
作者：张少杰
日期：2021年10月26日
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random

# 设置中文字体，解决matplotlib无法显示中文字体的问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# dataset = pd.read_csv('dataSet/studentscores.csv')

x = range(2, 26, 2)
y = [12, 23, 25, 27, 28, 35, 30, 50, 20, 49, 59, 60]

# 设置图片大小 figsize（长，宽）， dpi 像素点
plt.figure(figsize=(20, 8), dpi=80)
# 绘图 plot画折线图
plt.plot(x, y)
# 画散点图
# plt.scatter(x, y)
# 画条形图
# plt.bar(x, y, width=0.3, color='orange')
# 画横的条形图
# plt.barh(x, y, height=1)
# plt.yticks(x, x)
# plt.xticks(range(min(y), max(y))[::5])
# 画柱状图
# plt.hist(y, 5)
# 绘制x轴的刻度
# plt.xticks(x:坐标轴的值，labels：与x一一对应的字符串，rotation= :旋转角度)
plt.xticks(x, x, rotation=270)
plt.yticks(range(min(y), max(y))[::5])
# 添加描述信息
plt.xlabel("时间")
plt.ylabel("温度 单位(℃)")
plt.title('10点到12点每分钟气温变化情况')
# 绘制网格 alpha透明度
plt.grid(alpha=0.5)
# 保存图片
# plt.savefig('./demo.jpg')
# 显示图片
plt.show()

# 实例训练一 绘画从11岁到31岁每年生病次数
# age = range(11, 31)
# gf_1 = [random.randint(1, 5) for i in range(11, 31)]
# gf_2 = [random.randint(1, 5) for i in range(11, 31)]
# plt.figure(figsize=(15, 15), dpi=100)
# plt.title('从11岁到31岁每年生病次数')
# plt.xlabel('年龄 单位：岁')
# plt.ylabel('生病次数 单位：次')
# x_labels = [f'{i}岁' for i in age]
# plt.xticks(age, x_labels)
# plt.yticks(range(0, 6))
# plt.plot(age, gf_1, label='自己', color='green')
# plt.plot(age, gf_2, label='同桌')
# # 添加图例 loc：图例的位置
# plt.legend(loc='upper right')
# plt.show()
