import numpy as np
import random
from collections import namedtuple, deque, defaultdict
import matplotlib.pyplot as plt
from mdp import *
import pdb
import util

BUFFER_SIZE = int(1e5)

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
		return (states, actions, rewards, next_states, dones)

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
			#pdb.set_trace()
			score += self.weights[f] * v
		return score

	# This algorithm will produce an action given a state.
	# Here we use the epsilon-greedy algorithm: with probability
	# |explorationProb|, take a random action.
	def getAction(self, state):
		self.numIters += 1
		if random.random() < self.explorationProb:
			return random.choice(self.actions(state))
		else:
			return max((self.getQ(state, action), action) for action in self.actions(state))[1]

	# Call this function to get the step size to update the weights.
	def getStepSize(self):
		return 1.0 / math.sqrt(self.numIters)

	# We will call this function with (s, a, r, s'), which you should use to update |weights|.
	# Note that if s is a terminal state, then s' will be None.  Remember to check for this.
	# You should update the weights using self.getStepSize(); use
	# self.getQ() to compute the current estimate of the parameters.
	def incorporateFeedback(self, state, action, reward, newState):
		# BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
		#print("max of None: {}".format(max([self.getQ(None, a) for a in self.actions(None)])))
		if newState is None:
			error = self.getQ(state, action) - reward
		else:
			error = self.getQ(state, action) - (reward + self.discount * max([self.getQ(newState, a) for a in self.actions(newState)]))
		error *= self.getStepSize()
		for f, v in self.featureExtractor(state, action, self.mdp):
			self.weights[f] = self.weights[f] - error * v
		# END_YOUR_CODE

def simpleFeatureExtractor(state, action, mdp):
	features = []
	features.append(('pos', state[1]))
	features.append(('speed', state[3]))
	features.append(('ttc', mdp._get_smallest_TTC(state)))
	return features



def qlearning(mdp, n_episodes=50000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):

	rl = QLearningAlgorithm(mdp.actions, mdp.discount(), simpleFeatureExtractor, mdp, 0.2)

	scores_window = deque(maxlen=100) # last 100 scores
	eps = eps_start
	for i_episode in range(1, n_episodes+1):
		s = mdp.startState()
		s = np.array(s) # convert tuple to np.array
		score = 0
		for t in range(max_t):
			#a = agent.act(s, eps)
			a = rl.getAction(s)

			#pdb.set_trace()
			sp, r = mdp.sampleSuccReward(s, a)
			if r < -1:
				pdb.set_trace()
			sp = np.array(sp) # convert tuple to np.array
			done = mdp.isEnd(sp)

			#agent.step(s, a, r, sp, done)
			rl.incorporateFeedback(s, a, r, sp)

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
qlearning(mdp)
