import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import pandas as pd
import torchvision.transforms as transforms
from LeNet import LeNet
from ResNet import ResNet18
import math
import os


def scaling(x, x_max=7461488.0, x_min=137.0):
    return float((x - x_min) / (x_max - x_min))


def unscaling(x, x_max=7461488.0, x_min=137.0):
    return x*(x_max-x_min)+x_min


def lgscaling(x, x_max=7461488.0, x_min=137.0):
    return float((lg(x) - lg(x_min)) / (lg(x_max) - lg(x_min)))


def unlgscaling(x, x_max=7461488.0, x_min=137.0):
    x = x*(lg(x_max)-lg(x_min))+lg(x_min)
    return unlg(x)


def lg(x):
    return math.log10(x)


def unlg(x):
    return 10**x


def get_images_and_values(data_name, dir_path):
    '''
    从图像数据集的根目录dir_path下获取所有类别的图像名列表和对应的标签名列表
    :param dir_path: 图像数据集的根目录
    :return: images_list, values_list
    '''

    images_list1 = []  # 文件名列表
    images_list2 = []  # 文件名列表
    values_list = []  # 标签列表
    data = pd.read_csv(data_name)
    # value_max = max(data['Count'])
    # value_min = min(data['Count'])
    # print(value_max)
    # print(value_min)

    for i in range(len(data)):

        img_path_1 = dir_path + '/' + str(data.iat[i, 0]) + '.tiff'
        img_path_2 = dir_path + '/' + str(data.iat[i, 1]) + '.tiff'
        images_list1.append(str(img_path_1))
        images_list2.append(str(img_path_2))
        # values_list.append(lg(data.iat[i, 2]))
        # values_list.append(scaling(data.iat[i, 5]))
        values_list.append(lgscaling(data.iat[i, 5]))

        # if data.iat[i, 2] > 1000:
        #     images_list.append(str(img_path))
        #     # values_list.append(lg(data.iat[i, 1]))
        #     values_list.append(scaling(data.iat[i, 1], value_max, value_min))
        #     # values_list.append(data.iat[i, 1])

    return images_list1, images_list2, values_list


def get_data_images_and_values(data_name, dir_path):
    '''
    从图像数据集的根目录dir_path下获取所有类别的图像名列表和对应的标签名列表
    :param dir_path: 图像数据集的根目录
    :return: images_list, values_list
    '''

    images_list1 = []  # 文件名列表
    images_list2 = []  # 文件名列表
    values_list = []  # 标签列表
    data = pd.read_csv(data_name, header=None, sep='\t')
    # value_max = max(data[6])
    # value_min = min(data[6])
    # print(value_max)
    # print(value_min)

    for i in range(len(data)):

        img_path_1 = dir_path + '/' + str(data.iat[i, 2]) + '_1.tiff'
        img_path_2 = dir_path + '/' + str(data.iat[i, 2]) + '_2.tiff'
        images_list1.append(str(img_path_1))
        images_list2.append(str(img_path_2))
        # values_list.append(lg(data.iat[i, 2]))
        # values_list.append(scaling(data.iat[i, 6]))
        values_list.append(lgscaling(data.iat[i, 6]))

        # if data.iat[i, 2] > 1000:
        #     images_list.append(str(img_path))
        #     # values_list.append(lg(data.iat[i, 1]))
        #     values_list.append(scaling(data.iat[i, 1], value_max, value_min))
        #     # values_list.append(data.iat[i, 1])

    return images_list1, images_list2, values_list

class MyDataset(Dataset):
    def __init__(self, data_name, dir_path, transform=None):
        self.data_name = data_name  # 数据集
        self.dir_path = dir_path  # 数据集根目录
        self.transform = transform
        self.images1, self.images2, self.values = get_data_images_and_values(self.data_name, self.dir_path)

    def __len__(self):
        # 返回数据集的数据数量
        return len(self.values)

    def __getitem__(self, index):
        img_path1 = self.images1[index]
        img_path2 = self.images2[index]
        value = self.values[index]
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
        sample = {'image1': img1, 'image2': img2, 'value': value}
        if self.transform:
            sample['image1'] = self.transform(sample['image1'])
            sample['image2'] = self.transform(sample['image2'])
        return sample


# 下面这个transform会把numpy array变成torch tensor，然后把rgb值归一到[0, 1]这个区间
transform = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.RandomVerticalFlip(p=0.5)
     ])

# torch.set_printoptions(threshold=np.inf)
train_dataset = MyDataset("data/data_info.txt", "data/data_sample", transform=transform)
# test_dataset = MyDataset("test.csv", "experiment_tp", transform=transform)

batchSize = 64

train_size = int(len(train_dataset) * 0.8)
test_size = len(train_dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

trainloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batchSize, shuffle=True)

# for index, batch_data in enumerate(trainloader):
#     print(batch_data['value'])
# for data, values in testloader:
#     print(data)

device = torch.device("cuda:0")
net = LeNet().to(device)
# ResNet18
# net = ResNet18().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_value = np.array([])

print("Start Training...")
for epoch in range(50):

    test_logit = np.array([])
    test_value = np.array([])
    ae = np.array([])

    net.train()
    for i, data in enumerate(trainloader):
        image1 = data['image1'].float()
        image2 = data['image2'].float()
        values = data['value'].float()
        values = torch.unsqueeze(values, dim=1)
        image1, image2, values = image1.to(device), image2.to(device), values.to(device) # 注意需要复制到GPU
        optimizer.zero_grad()
        outputs = net(image1, image2)
        loss = criterion(outputs, values)
        val = values.cpu().detach().numpy()
        out = outputs.cpu().detach().numpy()
        loss.backward()
        optimizer.step()

        # train_value = np.append(train_value, np.array(unscaling(val)))
    #     # train_logit = np.append(train_logit, np.array(unscaling(out)))
    # a_loss = criterion(torch.tensor(train_logit), torch.tensor(train_value))
    # if epoch % 1 == 0:
    #     print("train:"+str(a_loss))

    net.eval()  # 启动测试模式
    for i, data in enumerate(testloader):  # 输出验证集的平均误差
        image1 = data['image1'].float()
        image2 = data['image2'].float()
        values = data['value'].float()
        values = torch.unsqueeze(values, dim=1)
        image1, image2, values = image1.to(device), image2.to(device), values.cpu().detach().numpy()
        logits = net(image1, image2).cpu().detach().numpy()
        test_value = np.append(test_value, np.array(unlgscaling(values)))
        test_logit = np.append(test_logit, np.array(unlgscaling(logits)))
        # test_value = np.append(test_value, np.array(unscaling(values)))
        # test_logit = np.append(test_logit, np.array(unscaling(logits)))
        # value = np.append(value, np.array(values))
        # logit = np.append(logit, np.array(logits))

    average_loss = criterion(torch.tensor(test_logit), torch.tensor(test_value))
    loss_value = np.append(loss_value, np.array(average_loss))
    # print(np.around(logit, decimals=2))
    # print(np.around(value, decimals=2))
    # if epoch % 50 == 0:
    #     print(np.around(logit, decimals=2))
    #     print(np.around(value, decimals=2))
    #     print((logit-value)/value)
    if epoch % 1 == 0:
        print("test:"+str(average_loss))
        # print((logit - value) / value)

print("Done Training!")
torch.save(net, "model/LeNet-9.pt")

net.eval()  # 启动测试模式
test_logit = np.array([])
test_value = np.array([])

for i, data in enumerate(testloader):  # 输出验证集的平均误差
    image1 = data['image1'].float()
    image2 = data['image2'].float()
    values = data['value'].float()
    values = torch.unsqueeze(values, dim=1)
    image1, image2, values = image1.to(device), image2.to(device), values.cpu().detach().numpy()
    logits = net(image1, image2).cpu().detach().numpy()
    # test_value = np.append(test_value, np.array(unscaling(values)))
    # test_logit = np.append(test_logit, np.array(unscaling(logits)))
    test_value = np.append(test_value, np.array(unlgscaling(values)))
    test_logit = np.append(test_logit, np.array(unlgscaling(logits)))

ae = abs((test_logit-test_value)/test_value)
print(np.around(test_logit, decimals=2))
print(np.around(test_value, decimals=2))
print(ae)

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 密度图
# Draw Plot
plt.figure(figsize=(16, 10), dpi=80)
# sns.kdeplot(logit, shade=True, color="g", label="predict", alpha=.7)
# sns.kdeplot(value, shade=True, color="black", label="value", alpha=.7)
# sns.kdeplot(ae, shade=True, color="red", label="ae", alpha=.7)

plt.axes(xscale="log")

# plt.plot(test_value,ae)
plt.scatter(test_value, ae)

import time
time_tuple = time.localtime(time.time())

result_path = "result/"+str(time_tuple[1])+'.'+str(time_tuple[2])+'-'+str(time_tuple[3])+'.'+str(time_tuple[4])
os.makedirs(result_path)

pd.DataFrame(test_value).to_csv(result_path + "/value.csv")
pd.DataFrame(test_logit).to_csv(result_path + "/logit.csv")
pd.DataFrame(loss_value).to_csv(result_path + "/loss.csv")

# Decoration
plt.title('Density Plot of City Mileage by n_Cylinders', fontsize=22)
plt.legend()
plt.savefig(result_path + '/scantter.jpg')
plt.show()

