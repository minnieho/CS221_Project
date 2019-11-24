import collections
import os

### Model (MDP problem)

class TransportationMDP(object):
	def __init__(self, N=10, tram_fail=0.5, discount=1):
		self.N = N
		self.tram_fail = tram_fail
		self.gamma = discount

	def startState(self, s):
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

	def discount(self):
		return self.gamma

## Inference (Algorithms)

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
