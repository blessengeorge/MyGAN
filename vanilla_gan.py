import os, sys, torch
from train import train
import weight_init
from nets import Generator, Discriminator
import input_pipe as ip

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

input_size = 28*28
hidden_size = 500

G = Generator((100, 500, 28*28), 'relu')
D = Discriminator((28*28, 500, 1), 'relu')

G.apply(weight_init.init_weights_normal)
D.apply(weight_init.init_weights_normal)

sample_size = int(1e6)
x_dataset = ip.MNIST_dataset()

z_dataset = ip.mv_gaussian(0, 1, mv_size=100, sample_size =sample_size)
z_dataset = torch.randn(sample_size, 100)

cuda = torch.cuda.is_available()

print ("PID: ", os.getpid())
if (len(sys.argv) == 2):
    folder = sys.argv[1]
else:
    folder = raw_input("Folder name\n")
if (os.path.exists(folder)):
    os.system("rm -r {0}".format(folder))
os.system("mkdir {0}".format(folder))

train(G, D, x_dataset, z_dataset, folder, cuda)
