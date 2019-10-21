import numpy as np
import copy
import random
import math
import pdb


def get_dist(obj1, obj2):
	return math.sqrt((obj1[0]-obj2[0])**2 + (obj1[1]-obj2[1])**2)

# Transition with Constant Acceleration model
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


# -------------
# Model
# -------------
# we frame the problem as a search problem, without uncertainties (cf CS221 W3 Search)
# - driving models are known
# - sensors are perfect (we know exactly the different state vectors, [x,y,vx,vy] for all cars)
# This is going to be used by our Oracle

class ActProblem(): # Anti Collision Tests problem
	# actions are accelerations
	#def __init__(self, nobjs=10, dist_collision=10, dt=0.25, actions=[-4., -2., -1., 0., +1., +2.]):
	def __init__(self, nobjs=10, dist_collision=10, dt=0.25, actions=[-2., -1., 0., +1., +2.]):
		self.nobjs = nobjs
		self.dist_collision = dist_collision
		self.dt = dt
		self.actions = actions
		# x, y, vx, vy
		self.start = np.array([100.0,	0.0,  0.0,		20.0], dtype=float)
		#self.goal  = np.array([100.0, 50.0, 0.0, 0.0], dtype=float) # down from 200 to 50
		self.goal  = np.array([100.0, 100.0, 0.0, 0.0], dtype=float) # down from 200 to 50
		self.start = self._randomStartState()

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
			cost = 1000
		elif abs(action) >= 2:
			cost = 2
		else:
			cost = 1
		return sp, cost

	def succAndCost(self, s):
		res = [] # (action, nextState, cost)
		for a in self.actions:
			sp, cost = self._step(s, a)
			res.append((a, tuple(sp), cost))
		return res

#random.seed(30)
#
#problem = ActProblem()
#start = problem.startState()
#print("start state: {}".format(start))
#print(problem.isEnd(start))

