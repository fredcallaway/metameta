from collections import namedtuple, defaultdict, deque, Counter
import numpy as np
from utils import PriorityQueue
from agents import Agent, Model
import gym
from gym import spaces
from policies import Policy, FixedPlanPolicy
from toolz import memoize, curry
import itertools as it

from utils import log_return
from distributions import *
# from envs import 

class MetaBestFirstSearchEnv(gym.Env):
    """A meta-MDP for best first search with a deterministic transition model."""
    Node = namedtuple('Node', ('state', 'path', 'reward', 'done'))
    State = namedtuple('State', ('frontier', 'reward_to_state', 'best_done'))
    TERM = 'TERM'

    def __init__(self, env, eval_node, expansion_cost=0.01):
        super().__init__()
        self.env = env
        self.expansion_cost = - abs(expansion_cost)

        # This guy interacts with the external environment, what a chump!
        self.surface_agent = Agent()
        self.surface_agent.register(self.env)
        self.eval_node = eval_node

    def _reset(self):
        self.env.reset()
        self.model = Model(self.env)  # warning: this breaks if env resets again
        start = self.Node(self.env._state, [], 0, False)
        frontier = PriorityQueue(key=self.eval_node(noisy=True))  # this is really part of the Meta Policy
        frontier.push(start)
        reward_to_state = defaultdict(lambda: -np.inf)
        best_done = None
        # Warning: state is mutable (and we mutate it!)
        self._state = self.State(frontier, reward_to_state, best_done)
        return self._state

    def _step(self, action):
        """Expand a node in the frontier."""
        if action is self.TERM:
            print('TERM')
            # The return of one episode in the external env is
            # one reward in the MetaSearchEnv.
            trace = self._execute_plan()
            external_reward = trace['return']
            return None, external_reward, True, {'trace': trace}
        else:
            return self._expand_node(action), self.expansion_cost, False, {}

    def _execute_plan(self):
        frontier, reward_to_state, best_done = self._state

        if not best_done:
            raise RuntimeError('Cannot make plan.')

        policy = FixedPlanPolicy(best_done.path)
        self.surface_agent.register(policy)
        print('execute')
        trace = self.surface_agent.run_episode(reset=False)
        return trace

        # elif frontier:
        #     plan = min(best_done, frontier.pop(), key=eval_node)
        #     plan = frontier.pop()

    def _expand_node(self, node):
        frontier, reward_to_state, best_done = self._state
        s0, p0, r0, _ = node

        for a, s1, r, done in self.model.options(s0):
            node1 = self.Node(s1, p0 + [a], r0 + r, done)
            if node1.reward <= reward_to_state[s1] -0.002:
                continue  # cannot be better than an existing node
            reward_to_state[s1] = node1.reward
            if done:
                best_done = max((best_done, node1), key=self.eval_node(noisy=False))
            else:
                frontier.push(node1)

        self._state = self.State(frontier, reward_to_state, best_done)
        return self._state


def heuristic(env, obs):
    row, col = obs
    g_row, g_col = env.goal
    return (abs(row - g_row) + abs(col - g_col))


# from models import BayesianRegression

class MetaBestFirstSearchPolicy(Policy):
    """Chooses computations in a MetaBestFirstSearchEnv."""
    def __init__(self, theta=None, n_iter=1):
        from models import BayesianRegression
        FEATURES = 2
        super().__init__()
        self.theta = theta
        if theta is None:
            self.n_iter = n_iter
            self.memory_length = 1000
            self.V = BayesianRegression(FEATURES)
            self.history = defaultdict(lambda: deque(maxlen=self.memory_length))

    def phi(self, node):
        # empty = not node.path
        reward_so_far = node.reward
        distance = heuristic(self.env.env, node.state)
        x = np.r_[reward_so_far, distance]
        return x

    @curry
    def eval_node(self, node, noisy):
        if node is None:
            return -np.inf
        elif self.theta:
            return self.theta @ self.phi(node)
        else:
            v, var = self.V.predict(self.phi(node), return_var=True)
            if noisy and self.i_episode <= 0:
                return v + np.random.randn() * var
            else:
                return v


    def finish_episode(self, trace):
        if self.theta is None:
            w, var = self.V.weights.get_moments()
            self.save('weights', w)
            self.save('weights_var', var)
            X = [self.phi(node) for node in trace['actions'][:-1]]
            y = list(reversed(np.cumsum(list(reversed(trace['rewards'])))))[:-1]

            self.history['X'].extend(X)
            self.history['y'].extend(y)

            X = np.stack(self.history['X'])
            y = np.array(self.history['y'])#.reshape(-1, 1)
            self.V.fit(X, y)

            # self.save('w', self.V.w)
            # X = np.array([self.phi(node) for node in trace['actions'][:-1]])
            # y = list(reversed(np.cumsum(list(reversed(trace['rewards'])))))
            # y = np.array(y[:-1]).reshape(-1, 1)
            # self.save('X', X)
            # self.save('y', y)
            # self.V.update(X, y)

            # h = self.model.fit(X, y, batch_size=len(X), epochs=1, verbose=0)
            # loss = h.history['loss']
            # from keras import backend as K
            # with K.get_session().as_default():
            #     self.save('weights', self.model.weights[0].eval())
            # self.save('loss', loss)
            

    def act(self, state):
        frontier, reward_to_state, best_done = state
        # print('frontier', frontier)
        # self.save('frontier', [n[1].state for n in frontier])
        self.ep_trace['frontier'].append([n[1].state for n in frontier])

        if best_done:
            value = self.eval_node(best_done, noisy=False)
            best = all(value > self.eval_node(n, noisy=False) 
                       for v, n in frontier)
            if best:
                return 'TERM'

        if frontier:
            return frontier.pop()
        else:
            print('NO FRONTIER')
            assert 0, 'no frontier'

      
