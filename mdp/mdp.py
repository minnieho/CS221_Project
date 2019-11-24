import collections
import os
import random
import math
import pdb

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

	def sampleSuccProbReward(self, s, a): # G(s,a) for mcts
		transitions = self.succProbReward(s, a)
		if self.sort: transitions = sorted(transitions)
		# sample a random transition
		i = sample([prob for newState, prob, reward in transitions])
		sp, prob, r = transitions[i]
		return (sp, prob, r)

	def discount(self):
		return self.gamma

	def states(self):
		return range(1, self.N+1)

	def pi0(self, s):
		#print("pi0({})".format(s))
		return random.choice(self.actions(s))

	# TODO add piBaseline
