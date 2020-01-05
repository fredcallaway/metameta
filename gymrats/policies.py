from abc import abstractmethod
import numpy as np
from scipy import stats

from .core import Component
from .utils import softmax

class Policy(Component):
    """Chooses actions."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def act(self, state):
        """Returns an action to take in the given state."""
        pass

    def attach(self, agent):
        if not hasattr(agent, 'env'):
            raise ValueError('Must attach env before attaching policy.')
        super().attach(agent)


class FunctionPolicy(Policy):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def act(self, state):
        return self.policy(state)


class RandomPolicy(Policy):
    """Chooses actions randomly."""

    def act(self, state):
        return self.env.action_space.sample()


class MaxQPolicy(Policy):
    """Chooses the action with highest Q value."""
    def __init__(self, Q, epsilon=0.5, anneal=.95, **kwargs):
        super().__init__(**kwargs)
        self.Q = Q
        self.epsilon = epsilon
        self.anneal = anneal

    def act(self, state, anneal_step=0):
        epsilon = self.epsilon * self.anneal ** anneal_step
        if np.random.rand() < epsilon:
            return np.random.randint(self.env.n_actions)
        else:
            q = self.Q.predict(state)
            noise = np.random.random(q.shape) * .001
            return np.argmax(q + noise)


class MaxQSamplePolicy(Policy):
    """Chooses the action with highest sampled Q value."""
    def __init__(self, Q, **kwargs):
        super().__init__(**kwargs)
        self.Q = Q

    def act(self, state):
        q, sigma = self.Q.predict(state, return_var=True)
        q_samples = stats.norm(q, sigma).rvs()
        q = q.flat
        a = np.argmax(q_samples)
        a1 = np.argmax(q)
        self.save('max', a == a1)
        self.save('regret', q[a1] - q[a])
        if a == self.env.term_action:
            self.save('final', state)
        return a


class SoftmaxPolicy(Policy):
    """Samples actions from a softmax over preferences."""
    def __init__(self, preference=None, temp=1e-9, noise=1e-9):
        super().__init__()
        if preference is None:
            assert hasattr(self, 'preference')
        else:
            self.preference = preference
        self.temp = temp
        self.noise = noise

    def act(self, state):
        probs = self.action_distribution(state)
        probs += np.random.rand(len(probs)) * self.noise
        probs /= probs.sum()
        return np.random.choice(len(probs), p=probs)

    def action_distribution(self, state):
        return softmax(self.preferences(state), self.temp)

    def preferences(self, state):
        q = np.zeros(self.n_action) - 1e30
        for a in self.env.actions(state):
            q[a] = self.preference(state, a)
        return q


