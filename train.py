import input_pipe as ip
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import chain
import itertools

import time
import matplotlib.pyplot as plt

Ng = 5
Nd = 5

minibatch_size = 40
sub_sample_size = 100
sample_size = sub_sample_size * Ng

sample_size = 20

learning_rate = 2e-4  
optim_betas = (0.9, 0.999)

num_epochs = 500000
print_interval = 500

#	Just be sure that this is doing what it is supposed to do.
def zero_grad(*nets):
	for net in nets:
		net.zero_grad()

def train(G, D, x_dataset, z_dataset, folder, cuda=False):
	
	variable = ip.cuda_variable if cuda else ip.variable
	
	criterion = nn.BCELoss()
	d_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=optim_betas)
	g_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=optim_betas)

	if cuda:
		G.cuda()
		D.cuda()

	realData = ip.sampler(x_dataset, sample_size)
	np.savetxt("{0}/real_data".format(folder), realData)

	start = time.time()

	MNIST = ip.MNIST_dataset()
	np.savetxt("{0}/gen_data_{1}".format(folder, 0), MNIST[0])	

	shuffleStart = time.time()
	np.random.shuffle(x_dataset)
	shuffleStop = time.time()

	print("Shuffle Time : {0}".format(shuffleStop - shuffleStart))

	for epoch in range(num_epochs):

		for index, real_data in enumerate(ip.dataLoader(x_dataset, sample_size)):

			real_sample_size = real_data.shape[0]
			fake_sample_size = sample_size

			#------PRINT INTERVAL

			if (index % print_interval) == 0:
				print("Generated dataset_{0}{1}".format(chr(epoch + 65), index / print_interval))

				g_input = variable( ip.sampler(z_dataset, sample_size))
				gen_data = ip.to_numpy(G(g_input))    
				np.savetxt("{0}/gen_data_{1}{2}".format(folder, chr(epoch + 65), index/print_interval), gen_data)	
				stop = time.time()
				print("Done in {0} seconds".format(round(stop - start, 2)))
				start = time.time()


			#-------DISCRIMINATOR TRAINING

			zero_grad(D)

			d_target = variable ( torch.cat( [torch.ones(real_sample_size), 		# Output Target
				torch.zeros(fake_sample_size)]))
			real_data = variable( real_data)		# Input
			fake_data = G(variable( ip.sampler(z_dataset, fake_sample_size))).detach()
			d_input = torch.cat([real_data, fake_data])

			loss = criterion( D(d_input), d_target.view(-1, 1))        
			loss.backward()

			d_optimizer.step()

			#------GENERATOR TRAINING

			zero_grad(G)

			g_target = variable( torch.ones( fake_sample_size)) 					# Output target
			g_input	= variable( ip.sampler(z_dataset, fake_sample_size))			# Input 

			loss = criterion( D(G(g_input)), g_target.view(-1, 1))
			loss.backward()

			g_optimizer.step()


def train_bayes(G, D, x_dataset, z_dataset, folder, cuda=False):

	variable = ip.cuda_variable if cuda else ip.variable
	
	criterion = nn.BCELoss()
	d_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=optim_betas)
	g_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=optim_betas)

	if cuda:
		G.cuda()
		D.cuda()

	realData = ip.sampler(x_dataset, sample_size)
	np.savetxt("{0}/real_data".format(folder), realData)

	start = time.time()

	for epoch in range(num_epochs):
		zero_grad(G, D)

		#------PRINT INTERVAL

		if (epoch % print_interval) == 0:
			print("Generated dataset_{0}".format(epoch / print_interval))

			gen_data = np.array([]).reshape(-1,1)

			for index in range(Ng):
				g_input = variable( ip.sampler(z_dataset, sub_sample_size))
				g_output = ip.to_numpy(G(g_input)).reshape(-1,1)
				gen_data = np.concatenate([gen_data, g_output])

			np.savetxt("{0}/gen_data_{1}".format(folder, epoch/print_interval), gen_data)
			stop = time.time()
			print("Done in {0} seconds".format(round(stop - start, 2)))
			start = time.time()

		#-------DISCRIMINATOR TRAINING
		# Train a set of Nd discriminators against the average generator output

		d_target = variable ( torch.cat( [torch.ones(sub_sample_size), 		# Output Target
			torch.zeros(sub_sample_size)]))
		real_data = variable( ip.sampler(x_dataset, sub_sample_size))		# Input
		fake_data = G(variable( ip.sampler(z_dataset, sub_sample_size))).detach()
		d_input = torch.cat([real_data, fake_data])

		for index in range(Nd):
			loss = criterion( D(d_input), d_target.view(-1, 1))        
			loss.backward()

		d_optimizer.step()

		#------GENERATOR TRAINING

		g_target = variable( torch.ones( sub_sample_size)) 					# Output target
		g_input	= variable( ip.sampler(z_dataset, sub_sample_size))			# Input 

		for index in range(Ng):
			loss = criterion( D(G(g_input)), g_target.view(-1, 1))
			loss.backward()

		g_optimizer.step()

def train_bayes_bigan(G, D, E, x_dataset, z_dataset, folder, cuda=False):
	variable = ip.cuda_variable if cuda else ip.variable
	
	criterion = nn.BCELoss()
	d_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=optim_betas)
	g_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=optim_betas)
	e_optimizer = optim.Adam(E.parameters(), lr=learning_rate, betas=optim_betas)

	if cuda:
		G.cuda()
		D.cuda()
		E.cuda()

	realData = ip.sampler(x_dataset, sample_size)
	np.savetxt("{0}/real_data".format(folder), realData)

	start = time.time()

	for epoch in range(num_epochs):
		
		zero_grad(G, D, E)

		#------PRINT INTERVAL

		if (epoch % print_interval) == 0:
			print("Generated dataset_{0}".format(epoch / print_interval))

			gen_data = np.array([]).reshape(-1,1)

			for index in range(Ng):
				g_input = variable( ip.sampler(z_dataset, sub_sample_size))
				g_output = ip.to_numpy(G(g_input)).reshape(-1,1)
				gen_data = np.concatenate([gen_data, g_output])

			np.savetxt("{0}/gen_data_{1}".format(folder, epoch/print_interval), gen_data)
			stop = time.time()
			print("Done in {0} seconds".format(round(stop - start, 2)))
			start = time.time()

		#-------DISCRIMINATOR TRAINING

		# Train a set of Nd discriminators against the average generator output
		d_target = variable ( torch.cat( [torch.ones(sub_sample_size * 2), 							# Output Target
			torch.zeros(sub_sample_size * 2)]))

		x = variable( ip.sampler(x_dataset, sub_sample_size))
		e_x = E(x).detach()
		z = variable( ip.sampler(z_dataset, sub_sample_size))
		g_z = G(z).detach()

		real_data = torch.cat([x, e_x])
		fake_data = torch.cat([g_z, z])

		d_input = torch.cat([real_data, fake_data])

		for index in range(Nd):
			loss = criterion( D(d_input), d_target.view(-1, 1))        
			loss.backward()

		d_optimizer.step()

		#------ENCODER TRAINING

		e_target = variable(  torch.zeros( sample_size * 2))										# Output target

		x = variable( ip.sampler(x_dataset, sample_size))
		e_x = E(x)
		
		e_output = torch.cat([x, e_x])
		e_loss = criterion( D(e_output), e_target.view(-1, 1))
		e_loss.backward()
		e_optimizer.step()

		#------GENERATOR TRAINING

		g_target = variable(  torch.ones(sub_sample_size * 2))  									# Output target

		z = variable( ip.sampler(z_dataset, sub_sample_size))

		for index in range(Ng):

			g_z = G(z)
			g_output = torch.cat([g_z, z])			
			g_loss = criterion( D(g_output), g_target.view(-1, 1))
			g_loss.backward()

		g_optimizer.step()

