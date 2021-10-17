""" 用python设计第一个游戏 """
cont = 3
while cont > 0:
    temp = input("不妨猜一下小甲鱼现在心里想的是哪个数字：")
    guess = int(temp)

    if guess == 8:
        print("你是小甲鱼心里的蛔虫嘛？！")
        print("哼，猜中了也没奖励！")
        break
    else:
        if guess < 8:
            print("小了")
        else :
            print("大了")
        cont = cont-1
print("游戏结束，不玩啦！")
