import torch
import torch.nn as nn

def init_weights_normal(m, gain=1):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight.data, gain)
#        m.weight.data.normal_(0, 100)
        
def init_weights_identity(m):
    if (type(m) == nn.Linear):
        shape = m.weight.data.numpy().shape
        mx = max(shape)
        i = torch.eye(mx)
        m.weight.data.copy_(i[:shape[0], :shape[1]])
        m.bias.data.fill_(0)
