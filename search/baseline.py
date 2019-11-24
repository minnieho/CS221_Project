import numpy as np
import pdb

DECELERATE = 0 # index in actions [-2., -1., 0, 1., 2.]
ACCELERATE = 3 # index in actions [-2., -1., 0, 1., 2.]

def get_TTC(ego, obj, radius):
	x1, y1, vx1, vy1 = ego[0], ego[1], ego[2], ego[3]
	x2, y2, vx2, vy2 = obj[0], obj[1], obj[2], obj[3]

	a = (vx1 - vx2) **2 + (vy1 - vy2) **2
	b = 2 * ((x1 - x2) * (vx1 - vx2) + (y1 - y2) * (vy1 - vy2))
	c = (x1 - x2) **2 + (y1 - y2) **2 - radius **2

	if a == 0 and b == 0:
		if c == 0:
			return 0
		else:
			return np.inf

	if a == 0 and b != 0:
		t = -c / b
		if t < 0:
			return np.inf
		else:
			return t

	delta = b **2 - 4 * a * c
	if delta < 0:
		return np.inf

	t1 = (-b - np.sqrt(delta)) / (2 * a)
	t2 = (-b + np.sqrt(delta)) / (2 * a)
	if t1 < 0:
		t1 = np.inf

	if t2 < 0:
		t2 = np.inf

	return min(t1, t2)

def get_smallest_TTC(s, dist_collision):
	radius = dist_collision
	#ego = s[0:4]
	ego = np.zeros(4)

	smallest_TTC = np.Inf
	smallest_TTC_obj = -1

	#idx = 4
	idx = 0
	#for n in range(int((len(s)-4)/4)):
	for n in range(int(len(s)/4)):
		obj = s[idx:idx+4]
		TTC = get_TTC(ego, obj, radius)

		if TTC < smallest_TTC:
			smallest_TTC = TTC
			smallest_TTC_obj = n
		idx += 4

	#return smallest_TTC, smallest_TTC_obj
	return smallest_TTC

# simple rule based baseline
def getBaselineAction(s):
	ttc = get_smallest_TTC(s, 10)
	if ttc <= 10:
		return DECELERATE
	else:
		return ACCELERATE

