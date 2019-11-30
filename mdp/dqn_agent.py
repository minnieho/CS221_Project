import numpy as np
import random
from collections import namedtuple, deque
import pdb
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils_nn as utils
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 4 # 64
TARGET_UPDATE = 100 # update target network every ... TODO TO BE TUNED 1000 looks OK
LR = 5e-4


class ReplayBuffer:
	def __init__(self, buffer_size, batch_size, seed):
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
		self.seed = random.seed(seed)

	def add(self, state, action, reward, next_state, done):
		e = self.experience(state, action, reward, next_state, done)
		self.memory.append(e)

	def sample(self):
		experiences = random.sample(self.memory, k=self.batch_size)

		states, actions, rewards, next_states, dones = [], [], [], [], []
		for e in experiences:
			states.append(e.state)
			actions.append(e.action)
			rewards.append(e.reward)
			next_states.append(e.next_state)
			dones.append(e.done)
		states = torch.from_numpy(np.array(states)).float().to(device)
		actions = torch.from_numpy(np.array(actions)).long().to(device)
		rewards = torch.from_numpy(np.array(rewards)).float().to(device)
		next_states = torch.from_numpy(np.array(next_states)).float().to(device)
		dones = torch.from_numpy(np.array(dones)).float().to(device)

		return (states, actions, rewards, next_states, dones)

	def __len__(self):
		return len(self.memory)

class DNN(nn.Module):
	def __init__(self, inputs=44, outputs=5):
		super(DNN, self).__init__()
		self.fc1 = nn.Linear(inputs, 200)
		self.fc2 = nn.Linear(200, 200)
		self.fc3 = nn.Linear(200, outputs)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		# XXX constrain the qvalues in [0,1] ???
		return x

class CNN(nn.Module):
	def __init__(self, inputs=44, outputs=5):
		super(CNN, self).__init__()
		nfilters = 256
		nfc = 200
		# in_channels, out_channels, kernel_size, stride
		self.conv1 = nn.Conv1d( 1, nfilters,  4, stride=4)
		self.bn1 = nn.BatchNorm1d(nfilters)
		self.conv2 = nn.Conv1d(nfilters, nfilters, 1, stride=1)
		self.bn2 = nn.BatchNorm1d(nfilters)
		self.maxpool = nn.MaxPool1d(10)
		self.fc1 = nn.Linear(4+nfilters, nfc)
		self.fc2 = nn.Linear(nfc, nfc)
		self.fc3 = nn.Linear(nfc, outputs)

	def forward(self, inputs):
		#pdb.set_trace()
		x = inputs[:, 4:]  # [4, 40]
		x = x.unsqueeze(1) # [N=4, Cin=1, L=40]
		x = F.relu(self.bn1(self.conv1(x))) # [N, 32, 10]
		x = F.relu(self.bn2(self.conv2(x))) # [N, 32, 10]
		x = self.maxpool(x) # [N, 32, 1]

		x = x.view(x.shape[0], -1) # [N, 32]
		ego = inputs[:, 0:4] # [N, 4]
		enc = torch.cat((ego, x), 1) # [N, 36]

		x = F.relu(self.fc1(enc))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		# XXX constrain the qvalues in [0,1] ???
		return x



class Agent():
	def __init__(self, state_size, action_size, gamma, args, seed):
		self.state_size = state_size
		self.action_size = action_size
		self.seed = random.seed(seed)
		self.gamma = gamma
		self.iters = 0
		self.args = args

		# Q-Network
		if (args.restore is not None and "cnn" in args.restore) or args.nn == 'cnn':
			self.dqn_local = CNN(state_size, action_size).to(device)
			self.dqn_target = CNN(state_size, action_size).to(device)
		else: # default to dnn
			self.dqn_local = DNN(state_size, action_size).to(device)
			self.dqn_target = DNN(state_size, action_size).to(device)
		self.optimizer = optim.Adam(self.dqn_local.parameters(), lr=LR)

		if args.restore is not None:
			restore_path = os.path.join('models/', args.restore + '.pth.tar')
			print("Restoring parameters from {}".format(restore_path))
			utils.load_checkpoint(restore_path, self.dqn_local, self.optimizer)
			self.dqn_target.load_state_dict(self.dqn_local.state_dict())

		# Replay memory
		self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
		self.t_step = 0

	# return loss
	def step(self, s, a, r, sp, done):
		self.memory.add(s, a, r, sp, done)
		if len(self.memory) > BATCH_SIZE:
			experiences = self.memory.sample()
			return self.learn(experiences)
		else:
			return None

	# ACHTUNG: act returns action INDEX !!!
	def act(self, state, eps=0.):
		# tuple to np.array to torch, use float, add batch dim and move to gpu
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.dqn_local.eval()
		with torch.no_grad():
			qvalues = self.dqn_local(state)
		self.dqn_local.train()

		# XXX how to forbid/penalize forbidden actions ???
		if random.random() > eps:
			return np.argmax(qvalues.cpu().numpy())
		else:
			return random.choice(np.arange(self.action_size))

	def learn(self, experiences):
		states, actions, rewards, next_states, dones = experiences

		# unsqueeze to get [batch_dim, 1], then squeeze to get back to a [batch_dim] vector
		Q_expected = self.dqn_local(states).gather(1, actions.unsqueeze(1)).squeeze(1)

		Q_targets_next = self.dqn_target(next_states).max(1)[0].detach()
		Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

		loss = F.mse_loss(Q_expected, Q_targets)

		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()
		# clip loss ??? clip params ???
		# for param in dqn_local.parameters():
		# 	param.grad.data.clamp_(-1, 1)
		self.optimizer.step()

		# TODO: update target network
		self.iters += 1
		if self.iters % TARGET_UPDATE == 0:
			self.dqn_target.load_state_dict(self.dqn_local.state_dict())

		return loss

	def save(self, episode, mean_score, is_best=True):
		now = datetime.now()
		dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
		filename = self.args.nn+'Date'+dt_string+'Episode'+str(episode)+'Score'+"{:.2f}".format(mean_score)
		print("Save model {} with mean_score {}".format(filename, mean_score))
		utils.save_checkpoint({'episode': episode,
								'state_dict': self.dqn_local.state_dict(),
								'optim_dict' : self.optimizer.state_dict(),
								'mean_score': mean_score},
								is_best = True,
								checkpoint = 'models/',
								filename=filename)

	def getV(self, state):
		# unsqueeze to add the BatchDim (1 in this case)
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.dqn_local.eval()
		with torch.no_grad():
			qvalues = self.dqn_local(state)
		qvalues = qvalues.cpu().numpy() # [BatchDim=1, 5]
		Vs = np.max(qvalues[0,:])
		return Vs

	def getQ(self, state, a): # a is an INDEX
		# unsqueeze to add the BatchDim (1 in this case)
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.dqn_local.eval()
		with torch.no_grad():
			qvalues = self.dqn_local(state)
		qvalues = qvalues.cpu().numpy() # [BatchDim=1, 5]
		Qsa = qvalues[0,a]
		return Qsa

