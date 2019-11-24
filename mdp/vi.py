import collections
import os
import random
import math
import pdb
from mdp import *


# -------------------
# Inference with VI
# -------------------
def valueIteration(mdp):
	V = collections.defaultdict(float)

	def Q(s,a):
		return sum([proba*(reward + mdp.discount()*V[sp]) for sp,proba,reward in mdp.succProbReward(s,a)])

	while True:
		newV = collections.defaultdict(float)
		for s in mdp.states():
			if mdp.isEnd(s):
				newV[s] = 0.
			else:
				newV[s] = max([Q(s,a) for a in mdp.actions(s)])
		if max([abs(V[s]-newV[s]) for s in mdp.states()]) < 1e-10:
			break
		V = newV

		# read out policy
		pi = {}
		for s in mdp.states():
			if mdp.isEnd(s):
				pi[s] = 'none'
			else:
				pi[s] = max([(Q(s,a), a) for a in mdp.actions(s)])[1]

		# print results
		os.system('clear')
		print('{:20} {:20} {:20}'.format('s', 'V(s)', 'pi(s)'))
		for s in mdp.states():
			print('{:>0} {:>20} {:>20}'.format(s, V[s], pi[s]))
		input()


#mdp = TransportationMDP(N=10, tram_fail=0.5)
#print(mdp.actions(3))
#print(mdp.succProbReward(3, 'walk'))
#print(mdp.succProbReward(3, 'tram'))
mdp = TransportationMDP(N=10)
valueIteration(mdp)
