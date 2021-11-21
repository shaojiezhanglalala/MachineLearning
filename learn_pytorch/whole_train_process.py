# -*- coding:utf-8 -*-
"""
作者：张少杰
日期：2021年11月16日
"""
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from model import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10('../dataSet', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10('../dataSet', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
#   加载数据
train_dataLoader = DataLoader(train_data, batch_size=64)
test_dataLoader = DataLoader(test_data, batch_size=64)

# 创建神经网络模型
tudui = Tudui()

#   损失函数
loss_fn = nn.CrossEntropyLoss()

#   优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

#   设置训练网路的一些参数
#   记录训练次数
total_train_step = 0
#   记录测试次数
total_test_step = 0
#   训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter('logs_seq')

for i in range(epoch):
    print('------第{}轮训练开始------'.format(i+1))

    #   训练步骤开始
    tudui.train()
    for data in train_dataLoader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        #   优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 ==0:
            print('训练次数:{}, loss:{}'.format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    #   测试步骤
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataLoader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print('整体测试集上的loss：{}'.format(total_test_loss))
    print('整体测试集上的正确率：{}'.format(total_accuracy / len(test_data)))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy / len(test_data), total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tudui, 'tudui_{}.pth'.format(i))
    print('模型已保存')

writer.close()