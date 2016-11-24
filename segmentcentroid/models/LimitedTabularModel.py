
from .AbstractModel import AbstractModel
import numpy as np
import scipy.special
from sklearn.preprocessing import normalize

"""
Defines a linear logistic model
"""


class LimitedTabularModel(AbstractModel):

    def __init__(self,statedim, actiondim, unnormalized=False):
      
        self.theta_map = {}
        self.isTabular = True
        self.actionbias = np.random.choice(np.arange(0,actiondim))

        super(LimitedTabularModel, self).__init__(statedim, actiondim, discrete=True, unnormalized=unnormalized)

    #returns a probability distribution over actions
    def eval(self, s):
        if self.actiondim == 1:
            return self.evalBin(s)

        s = np.ravel(s)
        sp = self.state_to_tuple(s)

        result = np.zeros((self.actiondim,1))

        for i in range(0,self.actiondim):
            if (sp,i) not in self.theta_map:
                
                if i == self.actionbias:
                    init = 1.0
                else:
                    init = 0.02

                result[i] = init
                self.theta_map[(sp,i)] = init
            else:
                result[i] = max(self.theta_map[(sp,i)],0.02)

        #print(np.squeeze(result)/np.sum(np.array(result)))

        return np.squeeze(result)/np.sum(np.array(result))

    #returns a probability distribution over actions
    def evalBin(self, s):
        s = np.ravel(s)
        sp = self.state_to_tuple(s)

        result = np.zeros((2,1))

        for i in range(0,2):
            if (sp,i) not in self.theta_map:
                init = np.random.choice(np.arange(1,2))
                result[i] = init
                self.theta_map[(sp,i)] = init
            else:
                result[i] = self.theta_map[(sp,i)]

        #print(np.squeeze(result)/np.sum(np.array(result)))

        return (np.squeeze(result)/np.sum(np.array(result)))[1]


    def log_deriv(self, s, a):
        s = np.ravel(s)
        sp = self.state_to_tuple(s)
        return (sp,a)

    def state_to_tuple(self, s):
        return tuple([i for i in s])

    def descent(self, grad_theta, learning_rate):

        for k in self.theta_map:
            self.theta_map[k] = 0.02

        for obs in grad_theta:
            weight = obs[0]
            state = self.state_to_tuple(obs[1][0])
            action = obs[1][1]

            if (state,action) not in self.theta_map:
                self.theta_map[(state,action)] = max(weight, 0.02)

            self.theta_map[(state,action)] = max(weight,0.02)

        #project onto top 10
        most_important = [(self.theta_map[k],k) for k in self.theta_map]
        most_important.sort(reverse=True)
        for t in most_important[min(10, len(most_important)):-1]:
            self.theta_map[t[1]] = 0.02


    def visited(self, s):
        return (self.state_to_tuple(s) in set([k[0] for k in self.theta_map]))
