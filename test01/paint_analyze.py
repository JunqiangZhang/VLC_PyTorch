'''
PyTorch神经网络线性拟合可见光数据
调用 matplotlib 进行绘图分析
加入 save() restore_params() 模块
JunqiangZhang@tom.com
200190514
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# 数据准备
filename = 'H:\\Py\\VLC_20190512(PyTorch)\\Data\\20190519.csv'
names = ['A', 'B', 'C', 'X', 'Y']
data = pd.read_csv(filename, names=names, delim_whitespace=True)
# print(type(data))
# 分离数据
array = data.values
# print(type(array))
intensity_00 = np.array(array[:, 0:3], dtype=np.float32)
X_and_Y = np.array(array[:, 3:5], dtype=np.float32)
validation_size = 0.1
seed = 0


min_max_scaler = preprocessing.MinMaxScaler((-1, 1), 1)
intensity_01 = min_max_scaler.fit_transform(intensity_00)
# print(intensity_01)
intensity_train, intensity_test, X_and_Y_train, X_and_Y_test = \
    train_test_split(intensity_01, X_and_Y, test_size=validation_size, random_state=seed)

# print(type(intensity_00))
length = len(intensity_train)

# 超参数设定
feature = 3
hidden = 200
output = 2
num_epochs = 1000000
learning_rate = 0.001

# 画散点图(抽样后参加训练的样点分布)
# plt.figure()
# plt.scatter(X_train, Y_train)
# plt.xlabel('X_train')
# plt.ylabel('Y_train')
# plt.show()
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# net = Net(n_feature=feature, n_hidden=hidden, n_output=output)
# print(net)

def save():
    net1 = Net(n_feature=feature, n_hidden=hidden, n_output=output)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net1.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        # Convert numpy array to torch Variable
        inputs = Variable(torch.from_numpy(intensity_train))
        targets = Variable(torch.from_numpy(X_and_Y_train))

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        pred = net1(inputs)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 500 == 0:
            print('Epoch [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, loss.item()))
    torch.save(net1.state_dict(), 'net_params.pkl2')  # save only the parameters

def restore_params(a):
    # intensity_new0 = a
    # intensity_new0 = intensity_new0.astype(np.float32)
    # intensity_new1 = np.concatenate((intensity_train, intensity_new0), axis=0)
    # min_max_scaler = preprocessing.MinMaxScaler((-1, 1), 1)
    # intensity_train_minmax = min_max_scaler.fit_transform(intensity_new1)
    # intensity_new2 = intensity_train_minmax[[length]]
    intensity_new3 = torch.from_numpy(a)

    net2 = Net(n_feature=feature, n_hidden=hidden, n_output=output)

    net2.load_state_dict(torch.load('net_params.pkl2'))
    prediction = net2(Variable(intensity_new3)).data.numpy()
    print(prediction)
    # data_x = prediction[0, 0]
    # data_y = prediction[0, 1]
    # return ([data_x, data_y])



save()
# for i, intensity_test2 in enumerate(intensity_test):
#     restore_params(intensity_test2)
#
# print('*************************************************************************')
# print(X_and_Y_test)