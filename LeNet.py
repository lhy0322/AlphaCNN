import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    # 一般在__init__中定义网络需要的操作算子，比如卷积、全连接算子等等
    def __init__(self):
        super(LeNet, self).__init__()
        # Conv2d的第一个参数是输入的channel数量，第二个是输出的channel数量，第三个是kernel size
        self.conv1 = nn.Conv2d(1, 16, (5, 5), (5, 5))
        self.conv2 = nn.Conv2d(16, 32, (5, 5), (5, 5))
        # self.conv3 = nn.Conv2d(32, 64, (3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接
        self.fc1 = nn.Linear(32*6*6*2, 1)
        self.fc2 = nn.Linear(128, 64)
        # 回归任务最后一个全连接层输出数量是1
        self.fc3 = nn.Linear(64, 1)
        # 防止过拟合
        self.dropout = nn.Dropout(0.2)

    # forward这个函数定义了前向传播的运算，只需要像写普通的python算数运算那样就可以了
    def forward(self, x, y):
        # 第一张图cnn
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # 下面这步把二维特征图变为一维，这样全连接层才能处理
        x = x.view(-1, 32 * 6 * 6)

        # 第二张图cnn
        y = F.relu(self.conv1(y))
        y = self.pool(y)
        y = F.relu(self.conv2(y))
        y = self.pool(y)
        # 下面这步把二维特征图变为一维，这样全连接层才能处理
        y = y.view(-1, 32 * 6 * 6)

        cat = torch.cat([x, y], dim=1)

        cat = F.relu(self.fc1(self.dropout(cat)))

        return cat
