import os, sys, torch
from train import *
import weight_init
from nets import BayesGenerator, BayesDiscriminator, Encoder
import input_pipe as ip

G = BayesGenerator((1, 100, 1), 'relu')
D = BayesDiscriminator((1, 100, 1), 'relu')
E = Encoder((1, 100, 1), 'relu')

G.apply(weight_init.init_weights_normal)
D.apply(weight_init.init_weights_normal)
E.apply(weight_init.init_weights_normal)

sample_size = int(1e6)
x_dataset = ip.mix_gaussian(sample_size =sample_size)
z_dataset = ip.gaussian(0, 5, sample_size =sample_size)

cuda = torch.cuda.is_available()

print ("PID: ", os.getpid())
if (len(sys.argv) == 2):
    folder = sys.argv[1]
else:
    folder = raw_input("Folder name\n")
if (os.path.exists(folder)):
    os.system("rm -r {0}".format(folder))
os.system("mkdir {0}".format(folder))

train_bayes_bigan(G, D, E, x_dataset, z_dataset, folder, cuda)

