# -*- coding:utf-8 -*-

"""
作者：张少杰
日期：2021年09月27日
"""
import keyword
from decimal import Decimal

# res = 0
# rg = range(0, 101, 2)
# i = 0
# while i < 100:
#     if i in rg:
#         res += i
#     i+=1
# print(res)
# print(Decimal('1.1')+Decimal('2.2'))
# print(1.1+2.1)
#
# print(keyword.kwlist)
# name = 'zsj'
# print('标识', id(name))
# print('类型', type(name))
# print('值', name)
# print('二进制', 0b10)
# print('八进制', 0o10)
# print('十六进制', 0x10)
# print(1.1+2.2)

# for item in range(100, 1000):
#     temp = 0
#     for i in str(item):
#         temp += int(i)**3
#     if temp == item:
#         print(item)
#         break

# for i in range(1,10):
#     for j in range(1,i+1):
#         print(j,'*', i,'=', i*j, end='\t')
#     print()
# lst =  ['hello', 'world', 98]
# lst2 = list(['hello', 'world', 99])
# print(lst)
# print(type(lst))
# print(lst2)
# print(type(lst2))
# lst = [10, 90, 30, 80, 50, 60, 70, 100]
# print(lst)
# print(id(lst))
# lst.sort(reverse=True)
# new_lst = sorted(lst, reverse=True)
# print(lst)
# print(id(lst))
# print(new_lst)
# print(id(new_lst))
# del lst
# del new_lst
# 列表生成式
# lst = [i*i for i in range(1,10)]
# print(lst)


# scores = {'张三': 99, '李四': 100, '王五': 95}
# print(scores['张三'])
# print(scores.get('九九', 101))
# scores1 = dict(zhangsan=99, lisi=100, wangwu=60)
# print(scores1)
# scores2 = {}
# print(scores2)
# print(type(scores2))
# print('张三' in scores)
# print('liuliu' in scores1)
# print('zs' not in scores2)
# del scores['张三']
# print(scores)
# scores1.clear()
# print(scores1)
# scores1['陈柳'] = 98
# print(scores1)
# scores1['陈柳'] = 100
# print(scores1)
# print(scores.keys())
# print(type(scores.keys()))
# print(list(scores.keys()))
# print(scores.values())
# print(type(scores.values()))
# print(scores.items())
# print(type(scores.items()))
# print(list(scores.items()))
# for item in scores:
#     print(item, scores.get(item))
#
# fruits = ['xigua', 'hamigua', 'taozi']
# prices = [50, 20]
# 字典生成式
# d = {item.upper(): price for item, price in zip(fruits, prices)}
# print(d)


# tup = ('zhangsan', 'lisi', 'wangwu')
# print(tup)
# print(id(tup))
# tup = ('zhaoliu',)
# print(tup)
# print(id(tup))
# tup1 = tuple(('zhangsan', 'lisi', 'wangwu', 'zhaoliu'))
# print(tup1)
#
# t = (10, [20, 30], 40)
# print(t, type(t), id(t))
# print(t[0], type(t[0]), id(t[0]))
# print(t[1], type(t[1]), id(t[1]))
# print(t[2], type(t[2]), id(t[2]))
# t[1].append(100)
# print(t[1], type(t[1]), id(t[1]))
#
# for i in t:
#     print(i)
#
# s = {1, 1, 1, 2, 3, 5}
# print(s)
# s.add(6)
# print(s)
# s.update([7, 8, 9])
# print(s)
# s.remove(2)
# print(s)
# s.discard(2)
# print(s)
# s = set()
# print(s, type(s))
# s = set([1, 1, 2, 2, 4, 5])
# print(s)
# s.clear()
# print(s)

# a = {10, 20, 30, 40}
# b = {10, 30, 20, 50, 90, 100}
# c = {50}
# print(a != b)
# 判断子集
# print(a.issubset(b))
# 判断超集
# print(a.issuperset(b))
# 判断交集
# print(a.isdisjoint(b))
# print(b.isdisjoint(c))
# 交集
# print(a.intersection(b))
# print(a & b)
# print(a)
# print(b)

# 并集
# print(a.union(b))
# print(a | b)
# print(a)
# print(b)

# 差集
# print(a.difference(b))
# print(a - b)
# print(a)
# print(b)

# 对称差集
# print(a.symmetric_difference(b))
# print(a ^ b)

# 集合生成式
# s = {i*i for i in range(1, 10)}
# print(s)

# 字符串
# 驻留机制
# 长度为0,1的字符串
# 符合标识符的字符串 （只含字母数字下划线的字符串）
# 在编译时驻留，不在运行时
# -5 ~ 256的整数
# sys中的intern方法强制两个字符串指向同一个对象
# s = 'hello, hello'
# 查找子串第一次或最后一次出现的索引
# 子串不存在时会报错
# print(s.index('lo'))
# 不报错
# print(s.find('lo'))
# 报错
# print(s.rindex('lo'))
# 不报错
# print(s.rfind('lo'))
# 字符串大小写转换 转换后均为创建新的字符串,地址id改变
# s = 'hello, Abc'
# 转大写
# print(s.upper())
# 转小写
# print(s.lower())
# 交换大小写
# print(s.swapcase())
# s = 'abc, def'
# 第一个字符大写,后面字符均小写
# print(s.capitalize())
# 每个单词第一个字符大写,其余小写
# print(s.title())

# s = 'hello,python'
# 若宽度小于字符串本身长度则返回原字符串
# 居中
# print(s.center(20, '*'))
# 左对齐
# print(s.ljust(20, '*'))
# 右对齐
# print(s.rjust(20, '*'))
# 右填充,用0填充,只有一个参数,设置字符串长度
# print(s.zfill(20))
# 会在负号后面添加0
# print('-987'.zfill(8))

# 分割字符串，默认分隔符为空格，sep设置分隔符，maxsplit设置最大分割次数
# split从左侧分割，rsplit从右侧分割
# s = 'hello, python, come, up'
# lst = s.split()
# print(lst, type(lst))
# lst = s.split(maxsplit=1)
# print(lst)
# lst = s.split(sep='co')
# print(lst)

# 判断字符串是否符合标识符
# 标识符只含数字字母下划线
# print('zsj_0414'.isidentifier())
# 判断字符串是否全由空字符组成,制表符\回车
# print('\t'.isspace())
# 判断字符串是否全由字母自称
# print('aaabc'.isalpha())
# 判断字符串是否全由十进制数字组成 罗马数字\中文数字均不算
# print('18292'.isdecimal())
# 判断字符串是否全由数字组成
# print('一Ⅰ'.isnumeric())
# 判断字符串是否全由字母和数字(所有数字)组成
# print('a271bs一Ⅲ'.isalnum())

# 替换字符串 replace() 被替换字符串,替换字符串,最大替换次数
# s = 'hello, python, python,python'
# print(s.replace('python', 'java', 2))
# print(s.replace('python', 'java'))

# 连接字符串 join
# lst = ['hell0', 'python', 'come', 'on']
# print('@'.join(lst))

# 格式化字符串
# 方法一  %s字符串 %d/%i整数 %f浮点数
# name = '张少杰'
# age = 22
# print('我叫%s, 今年%d岁' % (name, age))
# 标识进度  小数点前表示宽度,小数点后表示保留小数位数
# print("%.3f" % 3.1415926)

# 方法二
# print('我叫{0}, 今年{1}岁'.format(name, age))
# 标识进度  .3表示总共三位数,.3f表示保留三位小数, 0表示占位符的顺序 小数点前表示数字的宽度
# print("{0:.3f}".format(3.1415926))

# 方法三 3.0以上才能用
# print(f'我叫{name}, 今年{age}岁')

# 字符串的编码和解码
# s = '举头望明月'
# 编码
# 在GBK这种编码格式中,一个中文占两个字节
# print(s.encode(encoding='GBK'))
# 在UTF-8这种编码格式中,一个中文占两个字节
# print(s.encode(encoding='UTF-8'))

# 解码 编码格式必须和解码格式一致,否则无法解码
# byt = s.encode(encoding='gbk')
# print(byt.decode(encoding='gbk'))
#
# byt = s.encode(encoding='utf-8')
# print(byt.decode(encoding='utf-8'))


# 函数
# 如果参数是不可变对象,在函数体的修改不会影响实参的值
# 如果参数是可变对象,实参有可能会变，也有可能不变，这取决于进行改变的操作。
# 可以通过在函数体内拷贝可变对象,防止可变对象的改变 例一
# 如果函数没有返回值 return可省略；若为一个,返回原类型；若为多个则返回元组；
# 函数定义默认值参数，函数定义时，给形参设置默认值，只有当默认值不符的时候才需要传递实参。 例二
# 当定义一个有默认值参数的函数时，有默认值的参数必须位于所有没默认值参数的后面
# 函数的参数定义 个数可变的位置参数 *arg 结果为元组， 个数可变的关键字参数 **arg，结果为字典。
# 个数可变的位置参数和关键字参数均只能有一个；若两者参数都有，则位置形参必须在关键字形参之前；
# 在函数调用时，将列表中的每个元素都转换为位置实参传入(*list);将字典中的键值对都转换为关键字实参传入(**dict);
# def func(a,b,*,c,d) 在*之后的参数，在函数调用时，只能采用关键字参数
# 函数定义时形参的顺序问题 位置参数>关键字参数>个数可变的位置参数>个数可变的关键字参数
# 例一
# def change(x, y):
#     x = 2
#     y = y[:]
#     y[0] = 100
#     return
#
#
# a = 3
# b = [1, 2, 3]
# change(a, b)
# print(a, b)


# 例二
# def fun(arg1, arg2=10):
#     print(arg1, arg2)
#
# fun(100)
# fun(30, 40)

# try except else 捕获异常
# 常见异常：1.ZeroDivisionError 除(或取模)零(所有数据类型)
#         2.IndexError 序列中没有此索引(index)
#         3.KeyError 映射中没有这个键
#         4.NameError 未声明/初始化对象(没有属性)
#         5.SyntaxError python语法错误
#         6.ValueError 传入无效的参数
# traceback:打印异常信息
# import traceback
# try:
#     a = int(input('请输入第一个整数：'))
#     b = int(input('请输入第二个整数：'))
#     c = a / b
# except ZeroDivisionError:
#     traceback.print_exc()
#     print('0不能做除数')
# except ValueError:
#     print('只能输入数字')
# else:
#     print('程序正常执行！')
#     print('结果为：%.3f' % c)
# finally:
#     print('无论程序是否执行异常都会执行代码')
# print('程序结束')

# 类和对象
# 类的名称命名应遵守驼峰规则
# python是动态语言，在创建对象之后，可以动态地绑定属性和方法  对象名.属性或对象名.方法
# 如果类属性不希望在类的外部被使用，属性前加__,将属性私有化
# python支持多继承，定义子类时，必须在其构造函数中调用父类的构造函数
# Object类： dir查看类所具有的属性和可使用的方法; _str_ 类似Java的toString()方法
# class Student:
#     # 类属性
#     native_place = '吉林'
#
#     # 初始化方法
#     def __init__(self, name, age):
#         self.name = name
#         self.__age = age
#
#     def set_age(self,age):
#         if 0 < age <= 120:
#             self.__age = age
#         else:
#             self.__age = 18
#
#     def get_age(self):
#         return self.__age
#
#     # 实例方法
#     def eat(self):
#         print(f'学生{self.name}在吃饭')
#
#     # 静态方法   使用@staticmethod
#     @staticmethod
#     def method():
#         print('我是静态方法')
#
#     # 类方法   使用@classmethod
#     @classmethod
#     def cm(cls):
#         print('我是类方法')
#
#
# stu1 = Student('张少杰', 22)
# print(stu1.name, stu1.get_age(), type(stu1), id(stu1))
#
# stu1.eat()
# Student.method()
# Student.cm()
# # 动态添加属性
# stu1.gender = '男'
# print(stu1.gender)
#
#
# 动态添加方法

# def show():
#     print('定义在类之外的，称为函数')
#
#
# stu1.show = show()
# stu1.show

# class Person(object):
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
#
#     def info(self):
#         print(self.name, self.age)
#
#
# class Student(Person):
#     def __init__(self, name, age, node):
#         super().__init__(name, age)
#         self.node = node
#
#     def info(self):
#         super().info()
#         print(self.node)
#
#
# class Teacher(Person):
#     def __init__(self, name, age, year):
#         super().__init__(name, age)
#         self.year = year
#
#
# stu = Student('zsj', 20, 10201)
# stu.info()

# 深拷贝与浅拷贝
# 浅拷贝：import copy  copy.copy() 对象包含的子对象内容并不拷贝，源对象和拷贝对象会引用同一个子对象
# 深拷贝：import copy  copy.deepcopy() 递归拷贝对象中包含的子对象，源对象和拷贝对象所有的子对象也不相同

# import copy
#
#
# class Cpu:
#     pass
#
#
# class Dist:
#     pass
#
#
# class Computer:
#     def __init__(self, cpu, dist):
#         self.cpu = cpu
#         self.disk = dist


# cpu = Cpu()
# cpu2 = Cpu()
# print(id(cpu), id(cpu2))
# dist = Dist()
# computer = Computer(cpu, dist)
# # 浅拷贝
# computer2 = copy.copy(computer)
# print(id(computer), id(computer.cpu), id(computer.disk))
# print(id(computer2), id(computer2.cpu), id(computer2.disk))
# # 深拷贝
# computer3 = copy.deepcopy(computer)
# print(id(computer3), id(computer3.cpu), id(computer3.disk))


# 文件操作
# r只读, w只写,a追加写,b以字节流的形式读写,配合rwa使用,+既读又写,与rwa搭配使用
# 文件打开后使用完必须使用close()释放资源
# read(size) 读取指定大小字节的内容 readline() 读一行 readlines 以行为单位读取文件,将其以列表形式输出
# writeline() 写一行 writelines 写多行,以列表的形式传入要写的内容
# tell() 返回文件指针的当前位置
# flush()将缓冲区的内容写入文件,但不关闭文件
# close()将缓冲区的内容写入文件,并关闭文件
# a = open('a.txt', 'a+', encoding='utf-8')
# print(a.write('Hello,World!'))
# # seek() 设置读写指针位置 使用a+时,先写会将读写指针移到最后,若直接读文件则内容为空,须重置读写指针
# print(a.tell())
# a.seek(0)
# print(a.readlines())
# a.close()

# 传输图片音频文件时要用比特流的形式传输
# copy1 = open('copy.png', 'rb')
# copy2 = open('copy2.png', 'wb')
# copy2.write(copy1.read())
# copy1.close()
# copy2.close()

# with语句(上下文管理器),自动管理上下文资源,无论什么原因跳出with块,都能确保文件的正确关闭,从而释放资源
# 上下文管理器:类对象有特定的_enter_和_exit_方法,其实例对象就称为上下文管理器
# 不需要再写close语句
# with open('a.txt', 'r') as file:
#     print(file.read())

# os模块
# import os
# os.system('notepad.exe')
# os.system('calc.exe')
# 获取当前工作目录
# cwd = os.getcwd()
# print(cwd)
# # 返回指定路径下的目录和文件信息
# lst = os.listdir(cwd)
# print(lst)
# # 创建目录
# os.mkdir('NewDir')
# # 创建多级目录
# os.makedirs('NewDir2/b/c')
# # 删除目录
# os.rmdir('NewDir')
# # 删除多级目录
# os.removedirs('NewDir2/b/c')
# 设置当前工作目录
# os.mkdir('newDir')
# os.chdir('newDir')
# print(os.getcwd())
# os.rmdir('newDir')
# 递归遍历当前目录
import os
# import os.path
# cwd = os.getcwd()
# lst_file = os.walk(cwd)
# # dirPath为当前递归遍历的目录路径名, DirName当前遍历路径目录下所有的目录名, filename当前遍历路径目录下所有文件名
# for dirPath, DirName, filename in lst_file:
#     print(dirPath)
#     for dirName in DirName:
#         print(os.path.join(dirPath, dirName))
#     for fName in filename:
#         print(os.path.join(dirName, fName))
#     print('------------------------------------')

# os.path模块
# import os.path
# # 获取文件或目录的绝对路径
# absPath = os.path.abspath('one.py')
# print(absPath)
# # 判断文件或目录是否存在
# isExit = os.path.exists('one.py')
# print(isExit)
# isExit2 = os.path.exists('two.py')
# print(isExit2)
# # 将目录与目录或文件名拼接起来
# joinPath = os.path.join('D:\\One', 'one.py')
# print(joinPath)
# # split 分离路径名和文件名 splitext 分离文件名和扩展名
# sName = os.path.split('D:\\One\\one.py')
# print(sName)
# stName = os.path.splitext('one.py')
# print(stName)
# # 从目录中提取文件名
# basename = os.path.basename('D:\\One\\one.py')
# print(basename)
# # 从路径名中提取文件路径,不包括文件名
# dirName = os.path.dirname('D:\\One\\one.py')
# print(dirName)
# # 判断是否为路径
# isDir = os.path.isdir('D:\\One\\one.py')
# print(isDir)

# import numpy
# print('----------numpy.array-----------')
# a = numpy.array([[1, 2, 3, 4], [1, 2, 3, 5]], dtype=complex)
# print(a)
# print('----------shape-----------')
# print(a.shape)
# a.shape = (4, 2)
# print()
# print(a)
# b = a.reshape(8, 1)
# print()
# print(b)
# print('-----------ndarray.ndim 返回数组的维数')
# print(a.ndim)
#
# print('---------numpy.dtype-------------')
# dt = numpy.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
# a = numpy.array([('zsj', 20, 99.1), ('zzz', 22, 100), ('zsk', 23, 2920)], dtype=dt)
# print(a)
#
# print('-----------等间隔数字数组-------------')
# a = numpy.arange(24)
# b = a.reshape(2, 4, 3)
# print(b)
