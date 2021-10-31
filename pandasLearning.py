# -*- coding:utf-8 -*-
"""
作者：张少杰
日期：2021年10月28日
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# #  一维数组创建 pd.Series(数组, index=)
# t1 = pd.Series(np.arange(10), index=list('abcdefghij'))
# print(t1)
# temp_dict = {'name': 'zsj', 'age': 20, 'tel': 10086}
# t2 = pd.Series(temp_dict)
# print(t2)
# print(t2.index)
# print(t2.values)
#
# # 读取数据 下面为读取csv文件
# data = pd.read_csv('dataSet/dogNames2.csv')
# print(data)
#
# # DataFrame index行索引 columns列索引
# df1 = pd.DataFrame(np.arange(12).reshape((3, 4)), index=list('abc'), columns=list('wxyz'))
# print(df1)
# d1 = {'name': ['xiaoming', 'xiaogang'], 'age': [21, 22], 'tel': [10086, 192022]}
# df2 = pd.DataFrame(d1)
# print(df2)
# df3 = pd.DataFrame(data)
# print(df3)
# print(df3.head(20))
# # DataFrame基础属性
# print(df3.shape)
# print(df3.dtypes)
# print(df3.ndim)
# print(df3.index)
# print(df3.columns)
# print(df3.values)
# print(df3.info())  # 展示df3的概览
# print(df3.describe())   #描述数据类型为数字的元素的性质
#
# #   DataFrame中排序的方法 默认升序排列, ascending=False为降序
# df4 = df3.sort_values(by='Count_AnimalName', ascending=False)
# print(df4)
# #   pandas取行或取列的注意点
# #   方括号写数字，表示取行，对行操作
# #   方括号写字符串，表示取列，队列进行操作
# print(df4[:10]['Row_Labels'])
# print('*'*100)
# #   df.loc() 使用标签索引
# print(df4.loc[[12368, 8417], ['Row_Labels', 'Count_AnimalName']])
# #   df.iloc()使用位置索引
# print(df4.iloc[[0, 1, 2], :2])
# print(df4[(df4['Count_AnimalName'] > 800) & (df4['Count_AnimalName'] < 1000)])
# print(df4[df4['Row_Labels'].str.len() < 2])
# #   判断当前数组是否存在NAN pd.isnull(t) pd.notnull(t)
# print(pd.isnull(df4))
# print(pd.notnull(df4))
# print(df4[pd.isnull(df4['Row_Labels'])])
# #   删除NaN所在的行列 dropna(axis=0, how='any或all', inplace='True '原地替换原数组)
# #   在NaN处填充数据 t.fillna(t.mean())
# print(df4['Count_AnimalName'].mean())

# #   常用统计方法
# file_path1 = 'dataSet/IMDB-Movie-Data.csv'
# df = pd.read_csv(file_path1)
# # print(df.info())
# #   获取平均评分
# print(df['Rating'].mean())
# #   导演人数
# print(len(df['Director'].unique()))
# #   统计电影类别
# print(df['Genre'])
# temp_list = df['Genre'].str.split(',').tolist()
# genre_list = list(set([j for i in temp_list for j in i]))
# print(genre_list)
# zero_df = pd.DataFrame(np.zeros((df.shape[0], len(genre_list))), columns=genre_list)
# #   给出每个电影出现分类的位置赋值为1
# for i in range(df.shape[0]):
#     zero_df.loc[i, temp_list[i]] = 1
# print(zero_df)
# #   统计每个分类电影的数量
# genre_count = zero_df.sum(axis=0)
# genre_count = genre_count.sort_values(ascending=False)
# print(genre_count)
# #   画图
# print(type(genre_count))
# x = genre_count.index
# y = genre_count.values
# plt.figure(figsize=(20, 8), dpi=80)
# plt.bar(x, y)
# plt.show()

# #   数据合并之jion 按照行连接
# df1 = pd.DataFrame(np.ones((2, 4)), index=['A', 'B'], columns=list('abcd'))
# df2 = pd.DataFrame(np.ones((3, 4)), index=['A', 'B', 'C'], columns=list('wxyz'))
# print(df1)
# print(df2)
# print(df2.join(df1))
# #   数据合并之merge 按照列连接 默认为inner 求并集  outer 交集，NaN补全 left：左连接， right：右链接
# df3 = pd.DataFrame(np.arange(16).reshape(4, 4), index=['A', 'B', 'C', 'D'], columns=list('abkl'))
# print(df1.merge(df3, on='b'))
# print(df1.merge(df3, on='a', how='outer'))

#   分组聚合 t.groupby()
# starbucks_file_path = 'dataSet/directory.csv'
# df = pd.read_csv(starbucks_file_path)
# print(df.head(1))
# # print(df.info())
# grouped = df.groupby(by='Country')
# print(grouped)

#   DataFrameGroupBy 返回的对象是可迭代的，里面的元素为元组（索引（分组的值），分组后的DataFrame）
#   可以进行遍历
# for i, j in grouped:
#     print(i)
#     print('-'*100)
#     print(j)
#     print('*'*100)
# country_count = grouped['Brand'].count()
# print(country_count['US'])
# print(country_count['CN'])
#
# china_data = df[df['Country'] == 'CN']
# print(china_data.info())
# grouped = df.groupby(by=[df['State/Province'], df['Country']]).count()[['Brand', 'City']]
# grouped1 = df.groupby(by='State/Province').count()['Brand']
# print(grouped)
# print(type(grouped))
# print(type(grouped1))
# print(grouped.index)

#   pandas的时间序列 start:开始时间 end:结束时间 freq:频率 D=天 M=月，10D=10天 periods 表示次数
print(pd.date_range(start='20171230', end='20180131', periods=10))
print(pd.date_range(start='20171230', end='20180131', freq='10D'))
