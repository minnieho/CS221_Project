import time
import util
import pdb
import sys
sys.setrecursionlimit(10000)


# ---------------------------
# Algo: Backtracking search
# ---------------------------
def backtrackingSearch(problem):
	best = {'cost': float('inf'), 'history' : None}

	def recurse(s, history, totalCost):
		if problem.isEnd(s):
			if totalCost < best['cost']:
				best['history'] = history
				best['cost'] = totalCost
			return totalCost
		for a, sp, cost in problem.succAndCost(s):
			recurse(sp, history+[(a, sp, totalCost + cost)], totalCost + cost)

	recurse(problem.startState(), [], 0)
	return best

# -------------
# Print
# -------------
def printBestPath(best):
	print("totalCost={}".format(best['cost']))
	for (a, sp, totalCost) in best['history']:
		print("a={}, sp={}, cost={}".format(a, sp, totalCost))

# ---------------------------
# Algo: Dynamic Programming
# ---------------------------
def dynamicProgramming(problem):
	futureCost = {}
	def recurse(s):
		if s in futureCost:
			return futureCost[s][0]
		if problem.isEnd(s):
			return 0
		futureCost[s] = min([(cost + recurse(sp), a, sp, cost) for a, sp, cost in problem.succAndCost(s)])

		return futureCost[s][0]

	minCost = recurse(problem.startState())

	# recover history
	history = []
	s = problem.startState()
	while not problem.isEnd(s):
		_, a, s, cost = futureCost[s]
		history.append((a, s, cost))
	return minCost, history

# ---------------------------
# Algo: Uniform Cost Search
# ---------------------------
def uniformCostSearch(problem):
	frontier = util.PriorityQueue()
	explored = {}
	previous = {}

	frontier.update(problem.startState(), 0)
	previous[problem.startState()] = None, None, 0
	while True:
		s, pastCost = frontier.removeMin()
		prev_s, prev_a, prev_cost = previous[s]
		explored[s] = (pastCost, prev_s, prev_a, prev_cost)
		if problem.isEnd(s):
			minCost = pastCost
			break
		for a, sp, cost in problem.succAndCost(s):
			if sp in explored:
				continue
			if frontier.update(sp, pastCost+cost):
				previous[sp] = s, a, cost

	#print("EXPLORED: ", explored)

	history = []
	##sate = end_state
	while s is not None:
		pastCost, prev_s, prev_a, prev_cost = explored[s]
		history.append((prev_a, s, prev_cost))
		s = prev_s
	history.reverse()
	history = (minCost, history)
	print("REVERSE HISTORY: ", history)



# -------------
# Sample Model
# -------------
# just to validate the algorithms
class TransportProblem():
	def __init__(self, N):
		self.end = N

	def startState(self):
		return 1

	def isEnd(self, s):
		return (s >= self.end)

	def succAndCost(self, s):
		res = [] # (action, nextState, cost)
		if s+1 <= self.end:
			res.append(('walk', s+1, 1))
		if 2*s <= self.end:
			res.append(('tram', 2*s, 2))
		return res

problem = TransportProblem(N=1000)

#print(problem.succAndCost(4))

# start = time.time()
# best = backtrackingSearch(problem)
# end = time.time()
# printBestPath(best)
# print("backtrackingSearch time: {}".format(end-start))

start = time.time()
print(dynamicProgramming(problem))
end = time.time()
print("dynamicProgramming time: {} sec".format(end-start))

start = time.time()
uniformCostSearch(problem)
end = time.time()
print("uniformCostSearch time: {} sec".format(end-start))
