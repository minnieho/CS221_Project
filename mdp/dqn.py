import numpy as np
from collections import deque
from mdp import *
from dqn_agent import *
import pdb

import argparse

def dqn(mdp, args, n_episodes=50000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):

	agent = Agent(mdp.state_size(), mdp.action_size(), mdp.discount(), args, seed=0)

	best_mean_score = -math.inf
	scores_window = deque(maxlen=100) # last 100 scores
	eps = eps_start
	for i_episode in range(1, n_episodes+1):
		s = mdp.startState()
		score = 0
		for t in range(max_t):
			a = agent.act(mdp.normalize_state(s), eps) # Acthung: a is an index in the action set !!!
			sp, r = mdp.sampleSuccReward(s, a, actionIndex=True) # BUG FIX !!! a is an index
			done = mdp.isEnd(sp)
			l = agent.step(mdp.normalize_state(s), a, r, mdp.normalize_state(sp), done) # Achtung: a is an index in the action set !!!
			ttc, _ = mdp._get_smallest_TTC(sp)
			score += r
			if done:
				break
			s = sp
		scores_window.append(score)
		mean_score = np.mean(scores_window)
		if i_episode > 100 and mean_score > best_mean_score:
			#agent.save(i_episode, mean_score)
			best_mean_score = mean_score
		eps = max(eps_end, eps_decay*eps)
		print("Episode {} Average sliding score: {:.2f}".format(i_episode, mean_score))

# run python3 dqn.py or python3 dqn.py --restore best or python3 dqn.py -nn cnn
parser = argparse.ArgumentParser()
parser.add_argument('--nn', default='dnn', help="dnn or cnn")
parser.add_argument('--restore', default=None, help="Optional, file in models containing weights to reload before training")

args = parser.parse_args()

mdp = ActMDP()
dqn(mdp, args)
