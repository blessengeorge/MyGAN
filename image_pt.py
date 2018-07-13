import numpy as np
import os
import re
import sys
import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torchvision.utils as utils
import torch

#os.system("sshpass -f \"/pass.txt\" scp -r george@gpu1.cse.iitk.ac.in:/data/blessen/thesis/GAN/genData ./")
if (len(sys.argv) == 2):
    folder = sys.argv[1]
else:
    folder = raw_input("Folder name\n")

if (os.path.exists("{0}/real_data".format(folder))):
    real_data = np.loadtxt("{0}/real_data".format(folder))

#if os.path.exists("{0}/plots".format(folder)):
#    for np_file in os.listdir("{0}/plots".format(folder)):


if (not os.path.exists("{0}/plots".format(folder)) ):
    os.mkdir("{0}/plots".format(folder))

gs = gridspec.GridSpec(4, 4)

for np_file in sorted(os.listdir(folder)):
    if (re.match('gen*', np_file)):
        gen_data = np.loadtxt("{0}/{1}".format(folder, np_file))
        gen_data = torch.Tensor(gen_data.reshape(-1, 1, 28, 28))
        print(np_file)

        fig = plt.figure(figsize=(4, 4))
        for grid in gs:
            plt.subplot(grid)
            plt.axis('off')
            randIndex = random.randint(1, gen_data.shape[0]-1)

            # print(gen_data[randIndex][0][0])
            plt.imshow(gen_data[randIndex].reshape(28, 28), cmap='Greys_r')         
            # plt.show()

        plt.savefig("{0}/plots/{1}_plot".format(folder, np_file))
        plt.close()

            # utils.save_image(gen_data[index], "{0}.jpg".format(index))

#         bins = np.linspace(-100, 100, 200)
#         plt.hist(real_data, bins=bins, normed=1, alpha=0.5, color='r')
#         plt.hist(gen_data, bins=bins, normed=1, alpha=0.5, color='g')
#         plt.grid(True)
#         # plt.axes = ([-4, 4, 0, 1])
#         value=0
#         for i in np_file.split('_'):
#             if i.isdigit():
#                 value= int(i)
#                 break

#         plt.title(r"$\bf{ Iteration: " + i + "}$")
# #        plt.show()
#         if (not os.path.exists("{0}/plots".format(folder)) ):
#             os.mkdir("{0}/plots".format(folder))
#         plt.savefig("{0}/plots/{1}_plot".format(folder, np_file))
#         plt.clf()
#         plt.cla()        

