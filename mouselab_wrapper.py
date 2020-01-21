import numpy as np
from mouselab import MouselabEnv
from distributions import Categorical
import gym
from gym import spaces

class MouseLabWrapperEnv(gym.Env):
    def __init__(self,N,cost_range):
        super(MouseLabWrapperEnv, self).__init__()

        self.N = N
        self.cost_range = cost_range
        self.num_node_values = 2 #we're hardcoding the number of possible node values, but not the values themselves
        self.action_space = spaces.Discrete(N) #one action for each non-root node plus a terminal action makes N-1 + 1 = N actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.N+self.num_node_values + 1,self.N), dtype=np.float32) #this is 
        self._terminal_action = 0 #Note that this differs from the convention in MouseLabEnv
        self.reset()
        
    def encode_state_as_obs(self,state):
        vals = np.array(env.reward_distribution.vals)
        belief_state = np.vstack([[0,0]] + [o.probs if isinstance(o,Categorical) 
                        else np.arange(len(vals))==np.argmin(np.abs(vals-o))
                        for o in state[1:]])
        #Belief states are encoded as a N x num_node_values matrix, 
        #in which the i,j-th entry is the probability that node i has value reward_distribution.vals[j]
        #for revealed nodes, this probability is 0 or 1
        #for unrevealed nodes, the probability is copied from reward_distribution.probs
        #note that this encoding does not inform the agent of the actual values in reward_distribution.vals
        return np.hstack([self.adjacency_matrix,belief_state,np.full(shape=[self.N,1],fill_value=self.cost)])

    def step(self,action):
        #an action is an integer: 0 means terminating, any non-zero action means revealing the value of that node
        assert self.action_space.contains(action)
        if action==self._terminal_action:
            _ , reward, done, _ = self.env.step(self.env.term_action)
            #This is necessary to get around the convention
        else:
            _ , reward, done, _ = self.env.step(action)
        return self.encode_state_as_obs(self.env._state), reward, done, {}    
    
    
    def reset(self):
        
        self.cost = np.random.uniform(*self.cost_range)
    
        p = np.random.uniform(.1, .9)
        x = (1-p) / p
        self.reward_distribution = Categorical([-1, x], [(1-p), p])
        #We can make the set of possible reward distributions larger, this is a first attempt

        self.env = MouselabEnv.new_erdos_renyi(self.N, reward=self.reward_distribution, cost=self.cost, simple_features = True)
        self.adjacency_matrix = np.zeros([len(self.env.tree),len(self.env.tree)])
        for vertex,edges in enumerate(self.env.tree):
            for neighbor in edges:
                self.adjacency_matrix[vertex,neighbor] = 1
                
    def render():
        return
    