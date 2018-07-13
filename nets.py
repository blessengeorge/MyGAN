import base_net as bn
import torch
import torch.nn as nn
import torch.nn.functional as F
size = (1, 100, 1)

# 	Vanilla Generator and Discriminator
#	3 Linear Hidden layers, Discriminator output is applied squashing function
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

class Encoder(bn.BaseNet):
	def __init__(self, size, activ_fn_name='relu'):
		super(Encoder, self).__init__(size, activ_fn_name)

	def forward(self, x):
		x = self.activ_fn(self.map1(x))
		x = self.activ_fn(self.map2(x))
		x = self.activ_fn(self.map3(x))
		return self.map4(x)

# 	Bayes Generator and Discriminator 
#	Adds dropout layer

class BayesGenerator(bn.BaseNet):
	def __init__(self, size, activ_fn_name='relu'):
		super(BayesGenerator, self).__init__(size, activ_fn_name)
		self.drop1 = nn.Dropout()
		self.drop2 = nn.Dropout()
		self.drop3 = nn.Dropout()

	def forward(self, x):
		x = self.activ_fn(self.map1(x))
		x = self.drop1(x)
		x = self.activ_fn(self.map2(x))
		x = self.drop2(x)
		x = self.activ_fn(self.map3(x))
		# x = self.drop3(x)
		return self.map4(x)

class BayesDiscriminator(bn.BaseNet):
	def __init__(self, size, activ_fn_name='relu'):
		super(BayesDiscriminator, self).__init__(size, activ_fn_name)
		self.drop1 = nn.Dropout()
		self.drop2 = nn.Dropout()
		self.drop3 = nn.Dropout()

	def forward(self, x):
		x = self.activ_fn(self.map1(x))
		x = self.drop1(x)
		x = self.activ_fn(self.map2(x))
		x = self.drop2(x)
		x = self.activ_fn(self.map3(x))
		# x = self.drop3(x)
		return F.sigmoid(self.map4(x))

# 	Bayes Generator and Discriminator 
#	Discriminator output is not squashed

class Bayes_Logan_Generator(bn.BaseNet):
	def __init__(self, size, activ_fn_name='relu'):
		super(Generator, self).__init__(size, activ_fn[activ_fn_name])
		self.drop1 = nn.Dropout()
		self.drop2 = nn.Dropout()
		self.drop3 = nn.Dropout()

	def forward(self, x):
		x = self.activ_fn(self.map1(x))
		x = self.drop1(x)
		x = self.activ_fn(self.map2(x))
		x = self.drop2(x)
		x = self.activ_fn(self.map3(x))
		x = self.drop3(x)
		return self.map4(x)

class Bayes_Logan_Discriminator(bn.BaseNet):
	def __init__(self, size, activ_fn_name='relu'):
		super(Discriminator, self).__init__(size, activ_fn_name)
		self.drop1 = nn.Dropout()
		self.drop2 = nn.Dropout()
		self.drop3 = nn.Dropout()

	def forward(self, x):
		x = self.activ_fn(self.map1(x))
		x = self.drop1(x)
		x = self.activ_fn(self.map2(x))
		x = self.drop2(x)
		x = self.activ_fn(self.map3(x))
		x = self.drop3(x)
		return self.map4(x)

# Bayes Generator and Discriminator based on BIGAN setup

