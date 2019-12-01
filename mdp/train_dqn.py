import numpy as np
from collections import deque
from mdp import *
from dqn_agent import *
import pdb

import argparse
import logging
import utils_nn as utils

def train_dqn(mdp, args, n_epochs=100, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):

	agent = Agent(7, mdp.action_size(), mdp.discount(), args, seed=0)
	#agent = Agent(mdp.state_size(), mdp.action_size(), mdp.discount(), args, seed=0)

	best_score = -math.inf

	mean_score = -math.inf
	avg_tr_loss = 0
	eps = eps_start
	iters = 0
	for num_epoch in range(n_epochs):
		tr_scores_window = deque(maxlen=100) # last 100 scores
		for num_s, s in enumerate(mdp.train()):
			score = 0
			for t in range(max_t):
				iters += 1
				a = agent.act(mdp.reduce_state(s), eps) # a is an index !!!
				#a = agent.act(mdp.normalize_state(s), eps) # a is an index !!!
				sp, r = mdp.sampleSuccReward(s, a, actionIndex=True) # a is an index !!!
				done = mdp.isEnd(sp)
				l = agent.step(mdp.reduce_state(s), a, r, mdp.reduce_state(sp), done) # a is an index !!!
				#l = agent.step(mdp.normalize_state(s), a, r, mdp.normalize_state(sp), done) # a is an index !!!
				if l is not None:
					avg_tr_loss += l.item()
				ttc, _ = mdp._get_smallest_TTC(sp)
				score += r
				if done:
					break
				s = sp
				if iters%100 == 99:
					logging.info("Epoch no {}: sample {} iter {} avg_tr_loss: {:0.4f} tr_mean_score: {:.2f}".format(num_epoch, num_s, iters, avg_tr_loss/100, mean_score))
					avg_tr_loss = 0
			tr_scores_window.append(score)
			mean_score = np.mean(tr_scores_window)
			eps = max(eps_end, eps_decay*eps)

		dev_scores_window = deque(maxlen=100) # last 100 scores
		for num_s, s in enumerate(mdp.dev()):
			#print(num_s)
			score = 0
			for t in range(max_t):
				a = agent.act(mdp.reduce_state(s), eps=0.) # a is an index !!!
				#a = agent.act(mdp.normalize_state(s), eps=0.) # a is an index !!!
				sp, r = mdp.sampleSuccReward(s, a, actionIndex=True) # a is an index !!!
				done = mdp.isEnd(sp)
				score += r
				if done:
					break
				s = sp
			dev_scores_window.append(score)
		dev_mean_score = np.mean(dev_scores_window)
		logging.info("Epoch no {}: dev_mean_score: {:.2f}".format(num_epoch, dev_mean_score))
		if dev_mean_score > best_score:
			agent.save(iters, dev_mean_score)
			best_score = dev_mean_score

	#best_mean_score = -math.inf
	#scores_window = deque(maxlen=100) # last 100 scores
	#eps = eps_start
	#for i_episode in range(1, n_episodes+1):
	#	s = mdp.startState()
	#	score = 0
	#	for t in range(max_t):
	#		a = agent.act(mdp.reduce_state(s), eps) # Acthung: a is an index in the action set !!!
	#		sp, r = mdp.sampleSuccReward(s, a, actionIndex=True) # BUG FIX !!! a is an index
	#		done = mdp.isEnd(sp)
	#		l = agent.step(mdp.reduce_state(s), a, r, mdp.reduce_state(sp), done) # Achtung: a is an index in the action set !!!
	#		ttc, _ = mdp._get_smallest_TTC(sp)
	#		score += r
	#		if done:
	#			break
	#		s = sp
	#	scores_window.append(score)
	#	mean_score = np.mean(scores_window)
	#	if i_episode > 100 and mean_score > best_mean_score:
	#		#agent.save(i_episode, mean_score)
	#		best_mean_score = mean_score
	#	eps = max(eps_end, eps_decay*eps)
	#	print("Episode {} Average sliding score: {:.2f}".format(i_episode, mean_score))

utils.set_logger('train.log')

# run python3 dqn.py or python3 dqn.py --restore best or python3 dqn.py -nn cnn
parser = argparse.ArgumentParser()
parser.add_argument('--nn', default='dnn', help="dnn or cnn")
parser.add_argument('--restore', default=None, help="Optional, file in models containing weights to reload before training")

args = parser.parse_args()

mdp = ActMDP()
train_dqn(mdp, args)
