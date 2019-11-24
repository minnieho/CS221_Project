import collections
import os
import random
import math
import pdb
from mdp import *

## Inference (Algorithms)

# -------------------
# Inference with MCTS
# -------------------
def mcts(mdp, depth=10, iters=10000, sort=False):
	Tree = set()
	Nsa = {}
	Ns = {}
	Q = {}
	c = 1. # param controlling amount of exploration

	def selectAction(s, d=10, iters=100):
		for _ in range(iters):
			simulate(s, d, mdp.pi0)
		return max([(Q[(s,a)], a) for a in mdp.actions(s)])[1] # argmax

	def simulate(s, d, pi0):
		if d == 0 or mdp.isEnd(s): # typo in book ?
			return 0
		if s not in Tree:
			for a in mdp.actions(s):
				Nsa[(s,a)], Ns[s], Q[(s,a)] =  0, 1, 0. # TODO use expert knowledge
			Tree.add(s)
			return rollout(s, d, pi0)

		a = max([(Q[(s,a)]+c*math.sqrt(math.log(Ns[s])/(1e-5 + Nsa[(s,a)])), a) for a in mdp.actions(s)])[1] # argmax
		sp, r = mdp.sampleSuccReward(s, a)
		q = r + mdp.discount() * simulate(sp, d-1, pi0)
		Nsa[(s,a)] += 1
		Ns[s] += 1
		Q[(s,a)] += (q-Q[(s,a)])/Nsa[(s,a)]
		return q

	def rollout(s, d, pi0):
		if d == 0 or mdp.isEnd(s): # typo in book ?
			return 0
		a = pi0(s)
		sp, r = mdp.sampleSuccReward(s, a)
		return r + mdp.discount() * rollout(sp, d-1, pi0)

	s = mdp.startState()
	step = 1
	while True:
		a = selectAction(s, depth, iters)
		#a = 'tram' # 'walk'
		sp, r = mdp.sampleSuccReward(s, a)
		print("Step {}: (s,a,r,sp)=({}, {}, {:.1f}, {})".format(step, s,a,r,sp))
		step += 1
		s = sp
		if mdp.isEnd(s) or step > 1000:
			break

#mdp = TransportationMDP(N=10, tram_fail=0.5)
#print(mdp.actions(3))
#print(mdp.succProbReward(3, 'walk'))
#print(mdp.succProbReward(3, 'tram'))
mdp = TransportationMDP(N=10)
mcts(mdp)
