import collections
import os
import random
import math
import pdb
import time
from mdp import *
import dqn_agent as dqn

import argparse

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
    print("depth is:", depth)

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
            if 'nnet' in args.mcts:
#                print("never comes here!")
                for a in mdp.actions(s):
                    Nsa[(s,a)], Ns[s], Q[(s,a)] =  1, 1, nnet.getQ(mdp.normalize_state(s), mdp.action_index(a))
            else:
#				print('no neural network')
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
        if 'nnet' in args.mcts:
#            print("never comes here!")
            return nnet.getV(mdp.normalize_state(s))
        else:
#			print('no neural network')
            a = pi0(s)
            sp, r = mdp.sampleSuccReward(s, a)
            return r + mdp.discount() * rollout(sp, d-1, pi0)

    s = mdp.startState()
    step = 1
    score = 0
    avg_time = 0
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
        score += r
        ttc, _ = mdp._get_smallest_TTC(sp)
        #print("Step {}: ttc={:.2f} time={:.2f} sec (s,a,r,sp)=({}, {}, {:.1f}, {})".format(step, ttc, end-start, s,a,r,sp))
        avg_time += end-start
        print("Step {}: ttc={:.2f} time={:.2f} sec score={:.2f} (a,r,s,sp)=({}, {:.2f}, {}, {})".format(step, ttc, end-start, score, a,r,s,sp))
        if r == -1:
            pdb.set_trace()
        step += 1
        s = sp
        if mdp.isEnd(s) or step > 1000:
            avg_time = avg_time / step
#			print("Length of Ns is: {}, sum of Ns is: {}".format(len(Ns), sum(Ns.values())))
#			print(Ns.values())
            #print("Average over {} steps: time={:.2f} sec, final score={:.2f}".format(step, avg_time, score))
            break
    return (step, avg_time, score, (a,r,s,sp), Ns)
#mdp = TransportationMDP(N=10, tram_fail=0.5)
#print(mdp.actions(3))
#print(mdp.succProbReward(3, 'walk'))
#print(mdp.succProbReward(3, 'tram'))

#mdp = TransportationMDP(N=10)

parser = argparse.ArgumentParser()
parser.add_argument('--mcts', default='vanilla', help="vanilla or nnet")
parser.add_argument('--nn', default='dnn', help="dnn or cnn")
parser.add_argument('--restore', default='dnn', help="File in models containing weights to load for mcts nnet")
args = parser.parse_args()

# if 'nn' in args.nn:
# 	args.mcts = 'nnet'
# 	args.restore = args.nn

# Initialize mdp model
mdp = ActMDP()
print("Loading dqn agent: models/{}.pth.tar ...".format(args.restore))
nnet: object = dqn.Agent(mdp.state_size(), mdp.action_size(), mdp.discount(), args, seed=0)
print("Loading dqn agent: done (not used yet)".format(args.restore))
#pdb.set_trace()

s = mdp.startState()
ttc, _ = mdp._get_smallest_TTC(s)
print("V(startState)={}, ttc={} speed={}".format(nnet.getV(mdp.normalize_state(s)), ttc, s[3]))
for action in mdp.actions(s):
    a = mdp.action_index(action)
    print("Q(startState, {})={}".format(action, nnet.getQ(mdp.normalize_state(s),a)))

# Start mcts and gather statistics
depths = list(range(4,30,4))
#depths = list(range(12,13,1))
results = []
for depth in depths:
    step, avg_time, score, (a,r,s,sp), Ns = mcts(mdp, depth=depth)
    results.append((step, avg_time, score, len(Ns), sum(Ns.values())))
    #print("Depth {}: num steps {}: avg time={:.2f} sec score={:.2f} (a,r,s,sp)=({}, {:.2f}, {}, {})".format(depth, step, avg_time, score, a, r, s, sp))

print('Depth, (Num Steps, Average Time, Final Score, number of Ns states visited, total visits)')
for i in range(len(depths)):
    print(depths[i], results[i])

# for key, value in Ns.item():
# 	if value > 100:
# 		print(key, value)


