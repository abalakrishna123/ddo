import numpy as np
from segmentcentroid.envs.GridWorldEnv import GridWorldEnv
from segmentcentroid.envs.SwitchedGridWorldEnv import SwitchedGridWorldEnv
from segmentcentroid.planner.value_iteration import ValueIterationPlanner
from segmentcentroid.tfmodel.GridWorldModel import GridWorldModel
from segmentcentroid.inference.HDQN import *
import tensorflow as tf
import copy

gmap = np.loadtxt('resources/GridWorldMaps/11x11-Rooms-r1.txt', dtype=np.uint8)
mmap = np.loadtxt('resources/GridWorldMaps/11x11-Rooms-Modes.txt', dtype=np.uint8)

m  = GridWorldModel(3, statedim=(gmap.shape[0],gmap.shape[1]))

demonstrations = 100

full_traj = []
vis_traj = []

print(gmap.shape)

for i in range(demonstrations):
    print("Traj",i)
    # g = SwitchedGridWorldEnv(copy.copy(gmap), copy.copy(mmap), noise=0.3)
    g = GridWorldEnv(copy.copy(gmap), noise=0.3)

    v = ValueIterationPlanner(g)
    traj = v.plan(max_depth=100)
    # g.visualizePlan(traj)
    print("Beg Traj")
    print(traj)
    print("End Traj")

    new_traj = []
    for t in traj:
        a = np.zeros(shape=(4,1))

        s = np.zeros(shape=(gmap.shape[0],gmap.shape[1]))

        a[t[1]] = 1

        s[t[0][0],t[0][1]] = 1
        #s[2:4,0] = np.argwhere(g.map == g.START)[0]
        #s[4:6,0] = np.argwhere(g.map == g.GOAL)[0]

        new_traj.append((s,a))

    full_traj.append(new_traj)
    vis_traj.extend(new_traj)

m.sess.run(tf.initialize_all_variables())

with tf.variable_scope("optimizer"):
    opt = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
    m.train(opt, full_traj, 100, 1000)

actions = np.eye(4)


# g = SwitchedGridWorldEnv(copy.copy(gmap), copy.copy(mmap), noise=0.1)
g = GridWorldEnv(copy.copy(gmap), noise=0.3)
g.generateRandomStartGoal()

for i in range(m.k):
    states = g.getAllStates()
    policy_hash = {}
    trans_hash = {}

    # For each option, making a table of the transition probabilities
    # and actions for each state in the gridworld to understand learned options
    # For each option:
    # policy_hash maps states to actions, trans_hash maps states to option termination
    # probabilities, for THAT option.
    for s in states:
        # Not a problem here because trajectories are defined to be 11 x 11 with one-hot for state filled, but in HDQN clearly
        # that is not the case
        t = np.zeros(shape=(gmap.shape[0],gmap.shape[1]))

        t[s[0],s[1]] = 1
        #t[2:4,0] = np.argwhere(g.map == g.START)[0]
        #t[4:6,0] = np.argwhere(g.map == g.GOAL)[0]


        l = [ np.ravel(m.evalpi(i, [(t, actions[j,:])] ))  for j in g.possibleActions(s)]

        if len(l) == 0:
            continue

        #print(i, s,l, m.evalpsi(i,ns))
        action = g.possibleActions(s)[np.argmax(l)]

        policy_hash[s] = action

        trans_hash[s] = np.ravel(m.evalpsi(i, [(t, actions[1,:])]))

    g.visualizePolicy(policy_hash, trans_hash, blank=True, filename="resources/results/exp_stuff"+str(i)+".png")

# Train meta-controller with learned options
hdqn = TabularHDQN(g, (gmap.shape[0],gmap.shape[1]), 4, m)
hdqn.train(5000, 50)