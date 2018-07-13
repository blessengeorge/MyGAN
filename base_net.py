import torch.nn as nn
import torch.nn.functional as F

activ_fn = {
	'relu': F.relu,
	'elu': F.elu,
	'sigmoid': F.sigmoid
}

class BaseNet(nn.Module):
	def __init__(self, size, activ_fn_name):
		super(BaseNet, self).__init__()
		self.map1 = nn.Linear(size[0], size[1])
		self.map2 = nn.Linear(size[1], size[1])
		self.map3 = nn.Linear(size[1], size[1])
		self.map4 = nn.Linear(size[1], size[2])
		self.activ_fn = activ_fn[activ_fn_name]

	def forward(self, x):
		pass

#model = nn.Sequential(nn.Linear(3,5), nn.Linear(5,3))

