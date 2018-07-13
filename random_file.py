import itertools
import math
import time

import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable

import input_pipe as ip
import os, sys
import numpy as np

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_dataset = dsets.MNIST(root='./MNIST_data/', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(784, 1024),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(1024, 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         out = self.model(x.view(x.size(0), 784))
#         out = out.view(out.size(0), -1)
#         return out


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(100, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 1024),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(1024, 784),
#             nn.Tanh()
#         )
    
#     def forward(self, x):
#         x = x.view(x.size(0), 100)
#         out = self.model(x)
#         return out

from nets import Generator, Discriminator

G = Generator((100, 500, 28*28), 'relu')
D = Discriminator((28*28, 500, 1), 'relu')

discriminator = D.cuda()
generator = G.cuda()

criterion = nn.BCELoss()
lr = 0.0002
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)


def train_discriminator(discriminator, images, real_labels, fake_images, fake_labels):
    discriminator.zero_grad()

    # real_outputs = discriminator(images.reshape(-1, 28*28))
    # real_loss = criterion(outputs, real_labels)
    # real_score = real_outputs
    
    # fake_outputs = discriminator(fake_images) 
    # fake_loss = criterion(outputs, fake_labels)
    # fake_score = fake_outputs

    labels = torch.cat([real_labels, fake_labels])
    inputs = torch.cat([images.reshape(-1, 28*28), fake_images])
    outputs = discriminator(inputs)
    loss = criterion(outputs, labels)
    d_loss = loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss, None, None

def train_generator(generator, discriminator_outputs, real_labels):
    generator.zero_grad()
    g_loss = criterion(discriminator_outputs, real_labels)
    g_loss.backward()
    g_optimizer.step()
    return g_loss

num_test_samples = 16
test_noise = Variable(torch.randn(num_test_samples, 100).cuda())


# size_figure_grid = int(math.sqrt(num_test_samples))
# fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6, 6))
# for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
#     ax[i,j].get_xaxis().set_visible(False)
#     ax[i,j].get_yaxis().set_visible(False)

# set number of epochs and initialize figure counter
num_epochs = 200
num_batches = len(train_loader)
num_fig = 0

print ("PID: ", os.getpid())
if (len(sys.argv) == 2):
    folder = sys.argv[1]
else:
    folder = raw_input("Folder name\n")
if (os.path.exists(folder)):
    os.system("rm -r {0}".format(folder))
os.system("mkdir {0}".format(folder))

start = time.time()
i = 0
for epoch in range(num_epochs):
    for n, (images, _) in enumerate(train_loader):
        images = Variable(images.cuda())
        real_labels = Variable(torch.ones(images.size(0)).cuda())
        
        # Sample from generator
        noise = Variable(torch.randn(images.size(0), 100).cuda())
        fake_images = generator(noise)
        fake_labels = Variable(torch.zeros(images.size(0)).cuda())
        
        # Train the discriminator
        d_loss, real_score, fake_score = train_discriminator(discriminator, images, real_labels, fake_images, fake_labels)
        
        # Sample again from the generator and get output from discriminator
        noise = Variable(torch.randn(images.size(0), 100).cuda())
        fake_images = generator(noise)
        outputs = discriminator(fake_images)

        # Train the generator
        g_loss = train_generator(generator, outputs, real_labels)

        if (n+1) % 100 == 0:
            print("Generated dataset_{0}".format((i)))

            test_images = ip.to_numpy(generator(test_noise))
            print(test_images.shape)
            np.savetxt("{0}/gen_data_{1}".format(folder, ((i))), test_images)
            stop = time.time()
            print("Done in {0} seconds".format(round(stop - start, 2)))
            start = time.time()         
            i = i+1

            
            # for k in range(num_test_samples):
            #     i = k//4
            #     j = k%4
            #     ax[i,j].cla()
            #     ax[i,j].imshow(test_images[k,:].data.cpu().numpy().reshape(28, 28), cmap='Greys_r')
            
            # plt.savefig('results/mnist-gan-%03d.png'%num_fig)
            # num_fig += 1
            # print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, ' 
            #       'D(x): %.2f, D(G(z)): %.2f' 
            #       %(epoch + 1, num_epochs, n+1, num_batches, d_loss.data[0], g_loss.data[0],
            #         real_score.data.mean(), fake_score.data.mean()))

# fig.close()            