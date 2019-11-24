import collections
import os
import random
import math
import pdb
import numpy as np
import copy


# Return i in [0, ..., len(probs)-1] with probability probs[i].
def sample(probs):
	target = random.random()
	accum = 0
	for i, prob in enumerate(probs):
		accum += prob
		if accum >= target: return i
	raise Exception("Invalid probs: %s" % probs)


### Model (MDP problem)

class TransportationMDP(object):
	def __init__(self, N=10, tram_fail=0.5, discount=1, sort=False):
		self.N = N
		self.tram_fail = tram_fail
		self.gamma = discount
		self.sort = sort

	def startState(self):
		return 1

	def isEnd(self, s):
		return s >= self.N

	def states(self):
		return range(1, self.N+1)

	def actions(self, s):
		results = []
		if s+1 <= self.N:
			results.append('walk')
		if 2*s <= self.N:
			results.append('tram')
		return results

	def succProbReward(self, s, a):
		# returns sp, proba=T(s,a,sp), R(s,a,sp) 
		results = []
		if a == 'walk':
			results.append((s+1, 1., -1.))
		elif a == 'tram':
			results.append((2*s, 1. - self.tram_fail, -2.))
			results.append((s, self.tram_fail, -2.))
		return results

	def sampleSuccReward(self, s, a): # G(s,a) for mcts
		transitions = self.succProbReward(s, a)
		if self.sort: transitions = sorted(transitions)
		# sample a random transition
		i = sample([prob for newState, prob, reward in transitions])
		sp, prob, r = transitions[i]
		return (sp, r)

	def discount(self):
		return self.gamma

	def states(self):
		return range(1, self.N+1)

	def pi0(self, s):
		#print("pi0({})".format(s))
		return random.choice(self.actions(s))

	# TODO add piBaseline


def get_dist(obj1, obj2):
	return math.sqrt((obj1[0]-obj2[0])**2 + (obj1[1]-obj2[1])**2)

# Transition with Constant Acceleration model
# TODO missing covar matrix: so far just the mean, no uncertainty
# sofar almost the same as search model (except negative reward instead of positive cost)
def transition_ca(s, a, dt):
	Ts = np.matrix([[1.0, 0.0, dt,	0.0],
					[0.0, 1.0, 0.0, dt],
					[0.0, 0.0, 1.0, 0.0],
					[0.0, 0.0, 0.0, 1.0]])
	Ta = np.matrix([[0.5*dt**2, 0.0],
					[0.0, 0.5*dt**2],
					[dt, 0.0],
					[0.0, dt]])
	return np.dot(Ts, s) + np.dot(Ta, a)


class ActMDP(object): # Anti Collision Tests problem
	# actions are accelerations
	#def __init__(self, nobjs=10, dist_collision=10, dt=0.25, actions=[-4., -2., -1., 0., +1., +2.]):
	def __init__(self, nobjs=10, dist_collision=10, dt=0.25, actions=[-2., -1., 0., +1., +2.], discount=1):
		self.nobjs = nobjs
		self.dist_collision = dist_collision
		self.dt = dt
		self.actions = actions
		# x, y, vx, vy
		self.start = np.array([100.0,	0.0,  0.0,		20.0], dtype=float)
		self.goal  = np.array([100.0, 200.0, 0.0, 0.0], dtype=float) # down from 200 to 50
		self.start = self._randomStartState()
		self.gamma = discount

	# stase is R44: 1 ego + 10 cars, 4 coordonates (x,y,vx,vy) each
	def _randomStartState(self):
		state = copy.deepcopy(self.start)
		for n in range(int(self.nobjs/2)):
			x = float(random.randint(0, 50))
			y = float(random.randint(25, 190))
			vx = float(random.randint(10, 25))
			vy = float(random.randint(0, 5))
			obj = np.array([x, y, vx, vy])
			state = np.append(state, obj)

		for n in range(int(self.nobjs/2)):
			x = float(random.randint(150, 200))
			y = float(random.randint(25, 190))
			vx = - float(random.randint(10, 25))
			vy = - float(random.randint(0, 5))
			obj = np.array([x, y, vx, vy])
			state = np.append(state, obj)
		return state

	def startState(self):
		return tuple(self.start) # to make it hashable

	def isEnd(self, s):
		return (s[1] >= self.goal[1])

	def _get_dist_nearest_obj(self, s):
		nobjs = int(len(s)/4 - 1)
		ego = s[0:4]

		dist_nearest_obj = math.inf
		num_nearest_obj = -1

		idx = 4
		# TODO rewrite this in a more pythonic way
		for n in range(nobjs):
				obj = s[idx:idx+4]
				dist = get_dist(ego, obj)

				if dist < dist_nearest_obj:
						dist_nearest_obj = dist
						num_nearest_obj = n
				idx += 4

		#return dist_nearest_obj, num_nearest_obj
		return dist_nearest_obj


	# CA model for the ego vehicle and CV model for other cars
	def _step(self, state, action):
		sp = np.zeros_like(self.start)

		s = state[0:4]
		a = np.array([0.0, action])
		sp[0:4] = transition_ca(s, a, self.dt)

		idx = 4
		for n in range(self.nobjs):
			s_obj = state[idx:idx+4]
			a_obj = np.array([0.0, 0.0]) # CV model so far
			sp[idx:idx+4] = transition_ca(s_obj, a_obj, self.dt)
			idx += 4

		dist_nearest_obj = self._get_dist_nearest_obj(sp)
		# collision or driving backward (negative speed)
		if dist_nearest_obj < self.dist_collision or sp[3] < 0:
			reward = -1000
		elif abs(action) >= 2:
			reward = -2
		else:
			reward = -1
		return sp, reward

	def actions(self, s):
		# TODO restrict action set in some cases
		return self.actions

	def succProbReward(self, s, a):
		# we can't return a list
		raise NotImplementedError("Continuous state space")

	def sampleSuccReward(self, s, a): # G(s,a) for mcts or Q-learning
		sp, r = self._step(s, a)

	def discount(self):
		return self.gamma

	def pi0(self, s):
		#print("pi0({})".format(s))
		return random.choice(self.actions(s))

#random.seed(30)
#
#problem = ActProblem()
#start = problem.startState()
#print("start state: {}".format(start))
#print(problem.isEnd(start))

