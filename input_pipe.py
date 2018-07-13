import torch
import numpy as np
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data

mix_gaussian_values = [
                        (43, 1),
                        (-3, 1),
                        (-40, 1)
                    ]    
mix_gaussian_weights = [ 0.2, 0.1, 0.7 ]

def gaussian(mu, sig, sample_size = 100):
    return np.random.normal(mu, sig, (sample_size, 1))

def mv_gaussian(mu, sig, mv_size, sample_size=100):
    return torch.randn(sample_size, mv_size).numpy()
    return np.random.normal(mu, sig, (sample_size, mv_size))

def uniform(lo, hi, sample_size = 100):
    return np.random.uniform(lo, hi, (sample_size, 1))

def noise(sample_size=100):
    return gaussian(0, 1, sample_size)

def MNIST_dataset(sample_size=60000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    dataset = datasets.MNIST(root='./MNIST_data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size= sample_size, shuffle=True)
    return list(data_loader)[0][0].reshape(-1, 28*28)

def mix_gaussian(mv_list=mix_gaussian_values, weights=mix_gaussian_weights, sample_size=100):
    x = np.ndarray(0)
    for i in range(len(mv_list)):
        x = np.concatenate((x, np.random.normal(mv_list[i][0], mv_list[i][1], int(weights[i]*sample_size*1.0))))
    np.random.shuffle(x)    
    return x.reshape(-1, 1)  


#   Data Loaders 

def sampler(data_set, sample_size=100):
    choices = np.random.choice(data_set.shape[0], sample_size, replace=False)
    return data_set[choices, :]

class dataLoader:
    def __init__(self, data_set, batch_size):
        self.data_set = data_set
        self.batch_size = batch_size
        self.count = 0
        self.data_size = data_set.shape[0]
        self.maxLimit = (self.data_size // batch_size) + 1

    def __iter__(self):
        return self

    def next(self):
        if (self.count > self.maxLimit):
            raise StopIteration
        else:
            self.count += 1
            return self.data_set[self.batch_size * (self.count-1) : self.batch_size * (self.count)]

#   Converters

def torchify(x):
    return torch.Tensor(x)

def variable(x):
    return Variable( torchify(x))

def cuda_variable(x):
    return Variable( torchify(x)).cuda()

def to_numpy(x):
    if x.is_cuda:
        return x.cpu().data.numpy()
    else:
        return x.data.numpy()

