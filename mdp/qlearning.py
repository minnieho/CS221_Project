import numpy as np
import random
from collections import namedtuple, deque, defaultdict
import matplotlib.pyplot as plt
from mdp import *
import pdb
import util

import json
import pprint

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 4
LR = 5e-4

class ReplayBuffer:
	def __init__(self, buffer_size, batch_size, seed):
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		self.seed = random.seed(seed)

	def add(self, state, action, reward, next_state, done):
		e = (state, action, reward, next_state, done)
		self.memory.append(e)

	def sample(self):
		experiences = random.sample(self.memory, k=self.batch_size)
		return experiences

	def __len__(self):
		return len(self.memory)

# Performs Q-learning.	Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
	def __init__(self, actions, discount, featureExtractor, mdp, explorationProb=0.2):
		self.actions = actions
		self.discount = discount
		self.featureExtractor = featureExtractor
		self.explorationProb = explorationProb
		self.weights = defaultdict(float)
		self.numIters = 0
		self.mdp = mdp

	# Return the Q function associated with the weights and features
	def getQ(self, state, action):
		score = 0
		for f, v in self.featureExtractor(state, action, self.mdp):
			score += self.weights[f] * v
		return score

	# This algorithm will produce an action given a state.
	# Here we use the epsilon-greedy algorithm: with probability
	# |explorationProb|, take a random action.
	def getAction(self, state, eps):
		self.numIters += 1
		#if random.random() < self.explorationProb:
		if random.random() < eps: # align qlearning and dqn exploration strategy
			return random.choice(self.actions(state))
		else:
			return max((self.getQ(state, action), action) for action in self.actions(state))[1]

	# Call this function to get the step size to update the weights.
	def getStepSize(self):
		#return 1.0 / math.sqrt(self.numIters)
		return LR

	# We will call this function with (s, a, r, s'), which you should use to update |weights|.
	# Note that if s is a terminal state, then s' will be None.  Remember to check for this.
	# You should update the weights using self.getStepSize(); use
	# self.getQ() to compute the current estimate of the parameters.
	def incorporateFeedback(self, state, action, reward, newState, done=False):
		if newState is None or done:
			error = self.getQ(state, action) - reward
		else:
			error = self.getQ(state, action) - (reward + self.discount * max([self.getQ(newState, a) for a in self.actions(newState)]))
		#print("error={}".format(error))
		#error = min(1, error)
		#error = max(-1, error)
		error *= self.getStepSize()
		for f, v in self.featureExtractor(state, action, self.mdp):
			self.weights[f] = self.weights[f] - error * v

	def dumpWeights(self):
		pprint.pprint(json.loads(json.dumps(self.weights)), weightsFile)
		#print(dict(self.weights))

def actFeatureExtractor(state, action, mdp):
	features = []
	pos, speed, ttc_info = state[1], state[3], mdp._get_smallest_TTC(state)

	ttc, nobj = ttc_info
	idx = 4+nobj*4
	ttcX, ttcY, ttcVx, ttcVy = state[idx:idx+4]
	ttcX, ttcY, ttcVx, ttcVy = ttcX/200, ttcY/200, ttcVx/30, ttcVy/30

	# NB: trying to play with these features. I had to lower donw the learning rate (cf LR)
	features.append(('ttcX', ttcX))
	features.append(('ttcY', ttcY))
	features.append(('ttcVx', ttcVx))
	features.append(('ttcVy', ttcVy))

	features.append(('ttcX2', ttcX**2))
	features.append(('ttcY2', ttcY**2))
	features.append(('ttcVx2', ttcVx**2))
	features.append(('ttcVy2', ttcVy**2))

	#features.append(('ttcR', 1 - math.exp(-ttc/100.)))
	#features.append(('speedR', 1 - abs((speed-20.)/20.)))

	# normalize features, otherwise it does not work at all
	ttc = min(ttc,100)
	pos, speed, ttc, action = pos/200, speed/30, ttc/100, action/2

	# raw features
	features.append(('pos', pos))
	features.append(('speed', speed))
	#features.append(('ttc', ttc))
	#features.append(('ttc'+str(int(ttc)), 1))
	features.append(('bias', 1))

	# quadratic features
	features.append(('pos2', pos**2))
	features.append(('speed2', speed**2))
	#features.append(('ttc2', ttc**2))

	# action feature
	#features.append((math.copysign(1,action), 1))
	#features.append(('action', math.copysign(1,action)))
	#features.append(('action'+str(action), 1))
	#features.append((action, 1))
	features.append(('action', action))
	features.append(('action2', action**2))

	idx = 0
	#for i in range(mdp.nobjs):
	for i in range(0):
		x, y, vx, vy = state[idx:idx+4]
		x, y, vx, vy = x/200, y/200, vx/30, vy/30 # normalize
		features.append(('xCar'+str(i),   x))
		features.append(('yCar'+str(i),   y))
		features.append(('vxCar'+str(i), vx))
		features.append(('vyCar'+str(i), vy))

		features.append(('x2Car'+str(i),   x**2))
		features.append(('y2Car'+str(i),   y**2))
		features.append(('vx2Car'+str(i), vx**2))
		features.append(('vy2Car'+str(i), vy**2))
		idx += 4

	return features



def qlearning(mdp, n_episodes=50000, max_t=1000, eps_start=0.2, eps_end=0.01, eps_decay=0.999):

	rl = QLearningAlgorithm(mdp.actions, mdp.discount(), actFeatureExtractor, mdp, 0.2)
	memory = ReplayBuffer(BUFFER_SIZE, batch_size=BATCH_SIZE, seed=0)

	scores_window = deque(maxlen=100) # last 100 scores
	eps = eps_start
	for i_episode in range(1, n_episodes+1):
		s = mdp.startState()
		score = 0
		for t in range(max_t):
			#a = agent.act(s, eps)
			a = rl.getAction(s, eps)

			#pdb.set_trace()
			sp, r = mdp.sampleSuccReward(s, a)
			done = mdp.isEnd(sp)

			#agent.step(s, a, r, sp, done)
			memory.add(s, a, r, sp, done)
			if len(memory) > BATCH_SIZE:
				samples = memory.sample()
				for sample in samples:
					state, action, reward, next_state, isDone = sample
					rl.incorporateFeedback(state, action, reward, next_state, isDone)
			else:
				rl.incorporateFeedback(s, a, r, sp, done)

			ttc = mdp._get_smallest_TTC(sp)
			#print("Step {}: ttc={:.5f} (a,r,sp)=({}, {:.5f}, {})".format(t, ttc, a,r,sp[0:4]))
			score += r
			if done:
				break
			s = sp
		scores_window.append(score)
		eps = max(eps_end, eps_decay*eps)
		avg_sliding_score = np.mean(scores_window)
		print("Episode {} Average sliding score: {:.2f}".format(i_episode, avg_sliding_score))
		if avg_sliding_score > -10:
			weightsFile.write("Episode {} Average sliding score: {:.2f}\n".format(i_episode, avg_sliding_score))
			rl.dumpWeights()


weightsFile = open("qlearning.weights", "a")
mdp = ActMDP()
qlearning(mdp)
