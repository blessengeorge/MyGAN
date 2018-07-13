from __future__ import division, print_function, unicode_literals

import os

import numpy as np
import torch
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import math

import torchvision.datasets as datasets
from torch.autograd import Variable

import base_net as bn
import torch
import torch.nn as nn
import torch.nn.functional as F
size = (1, 100, 1)

#   Vanilla Generator and Discriminator
#   3 Linear Hidden layers, Discriminator output is applied squashing function
class Generator(bn.BaseNet):
    def __init__(self, size, activ_fn_name='relu'):
        super(Generator, self).__init__(size, activ_fn_name)

    def forward(self, x):
        x = self.activ_fn(self.map1(x))
        x = self.activ_fn(self.map2(x))
        x = self.activ_fn(self.map3(x))
        return self.map4(x)

class Discriminator(bn.BaseNet):
    def __init__(self, size, activ_fn_name='relu'):
        super(Discriminator, self).__init__(size, activ_fn_name)

    def forward(self, x):
        x = self.activ_fn(self.map1(x))
        x = self.activ_fn(self.map2(x))
        x = self.activ_fn(self.map3(x))
        return F.sigmoid(self.map4(x))

from PIL import Image

batch_size = 100
num_epochs = 5
learning_rate = 0.01

input_size = 28*28
hidden_size = 500

train_dataset = datasets.MNIST(root='./MNIST_data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./MNIST_data', train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class MNet(nn.Module):  #Extend PyTorch's Module class
    def __init__(self, num_classes = 10):
        super(MNet, self).__init__()  #Must call super __init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

model = Generator((28*28, 500, 10), 'relu')

criterion = nn.CrossEntropyLoss()

def train():
    learning_rate = 0.01
    for epoch in range(num_epochs):
        learning_rate *= 0.8
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for i, (images, labels) in enumerate(train_loader):

            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)
            optimizer.zero_grad()

            outputs = model(images)
            error = criterion(outputs, labels)
            error.backward()
            optimizer.step()

            if (i+1) % (1200/batch_size) == 0:
                print (error.data[0], batch_size)

        # if (i+1)== 10*1000:
        #     break

def test():
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(test_loader):
        images = Variable(images.view(-1, 28*28))

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        if (i+1)%(1200/batch_size) == 0:
            print(correct, total)
            print('Accuracy: %d %%' % (100.0 * correct / total))

# train()
# test()
