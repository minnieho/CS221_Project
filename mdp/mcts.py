import collections
import os
import random
import math
import pdb
import time
from mdp import *

## Inference (Algorithms)

# -------------------
# Inference with MCTS
# -------------------
def mcts(mdp, depth=12, iters=100, explorConst=1.0, tMaxRollouts=200, reuseTree=True):
	Tree = set()
	Nsa = {}
	Ns = {}
	Q = {}
	c = explorConst # param controlling amount of exploration
	tMax = tMaxRollouts # max steps used to estimate Qval via rollout

	def selectAction(s, d, iters, c):
		for _ in range(iters):
			simulate(s, d, mdp.pi0)
		return max([(Q[(s,a)], a) for a in mdp.actions(s)])[1] # argmax

	def simulate(s, d, pi0):
		if mdp.isEnd(s):
			return 0
		if d == 0: # we stop exploring the tree, just estimate Qval here
			return rollout(s, tMax, pi0)
		if s not in Tree:
			for a in mdp.actions(s):
				Nsa[(s,a)], Ns[s], Q[(s,a)] =  0, 1, 0. # TODO use expert knowledge
			Tree.add(s)
			# use tMax instead of d: we want to rollout deeper
			return rollout(s, tMax, pi0)

		a = max([(Q[(s,a)]+c*math.sqrt(math.log(Ns[s])/(1e-5 + Nsa[(s,a)])), a) for a in mdp.actions(s)])[1] # argmax
		sp, r = mdp.sampleSuccReward(s, a)
		q = r + mdp.discount() * simulate(sp, d-1, pi0)
		Nsa[(s,a)] += 1
		Ns[s] += 1
		Q[(s,a)] += (q-Q[(s,a)])/Nsa[(s,a)]
		return q

	def rollout(s, d, pi0):
		if d == 0 or mdp.isEnd(s):
			return 0
		a = pi0(s)
		sp, r = mdp.sampleSuccReward(s, a)
		return r + mdp.discount() * rollout(sp, d-1, pi0)

	s = mdp.startState()
	step = 1
	while True:
		if reuseTree is False:
			pdb.set_trace()
			Tree = set()
			Nsa = {}
			Ns = {}
			Q = {}
		start = time.time()
		a = selectAction(s, depth, iters, c)
		end = time.time()
		#a = 'tram' # 'walk'
		sp, r = mdp.sampleSuccReward(s, a)
		ttc = mdp._get_smallest_TTC(sp)
		#print("Step {}: ttc={:.2f} time={:.2f} sec (s,a,r,sp)=({}, {}, {:.1f}, {})".format(step, ttc, end-start, s,a,r,sp))
		print("Step {}: ttc={:.2f} time={:.2f} sec (a,r,s,sp)=({}, {:.2f}, {}, {})".format(step, ttc, end-start, a,r,s,sp))
		if r == 0:
			pdb.set_trace()
		step += 1
		s = sp
		if mdp.isEnd(s) or step > 1000:
			break

#mdp = TransportationMDP(N=10, tram_fail=0.5)
#print(mdp.actions(3))
#print(mdp.succProbReward(3, 'walk'))
#print(mdp.succProbReward(3, 'tram'))

#mdp = TransportationMDP(N=10)
mdp = ActMDP()
mcts(mdp)
