import numpy as np
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from mdp import *
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 4 # 64
TARGET_UPDATE = 10 # update target network every ...
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

# DQN: start experiments with a simple DNN
class DQN(nn.Module):
	def __init__(self, inputs=44, outputs=5):
		super(DQN, self).__init__()
		self.fc1 = nn.Linear(inputs, 100)
		self.fc2 = nn.Linear(100, 100)
		self.fc3 = nn.Linear(100, outputs)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		# XXX constrain the qvalues in [0,1] ???
		return x


class Agent():
	def __init__(self, state_size, action_size, gamma, seed):
		self.state_size = state_size
		self.action_size = action_size
		self.seed = random.seed(seed)
		self.gamma = gamma
		self.iters = 0

		# Q-Network
		self.dqn_local = DQN(state_size, action_size).to(device)
		self.dqn_target = DQN(state_size, action_size).to(device)
		self.optimizer = optim.Adam(self.dqn_local.parameters(), lr=LR)

		# Replay memory
		self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
		# Initialize time step (for updating every UPDATE_EVERY steps)
		self.t_step = 0

	def step(self, s, a, r, sp, done):
		self.memory.add(s, a, r, sp, done)
		if len(self.memory) > BATCH_SIZE:
			experiences = self.memory.sample()
			self.learn(experiences)

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
		#pdb.set_trace()

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


def dqn(mdp, n_episodes=500, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):

	agent = Agent(mdp.state_size(), mdp.action_size(), mdp.discount(), seed=0)

	scores_window = deque(maxlen=100) # last 100 scores
	eps = eps_start
	for i_episode in range(1, n_episodes+1):
		s = mdp.startState()
		s = np.array(s) # convert tuple to np.array
		score = 0
		for t in range(max_t):
			a = agent.act(s, eps)
			sp, r = mdp.sampleSuccReward(s, a)
			sp = np.array(sp) # convert tuple to np.array
			done = mdp.isEnd(sp)
			agent.step(s, a, r, sp, done)

			ttc = mdp._get_smallest_TTC(sp)
			#print("Step {}: ttc={:.5f} (a,r,sp)=({}, {:.5f}, {})".format(t, ttc, a,r,sp[0:4]))
			score += r
			if done:
				break
			s = sp
		scores_window.append(score)
		eps = max(eps_end, eps_decay*eps)
		print("Episode {} Average sliding score: {:.2f}".format(i_episode, np.mean(scores_window)))


mdp = ActMDP()
dqn(mdp)
