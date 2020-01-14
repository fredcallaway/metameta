from gymrats.policies import Policy
from bayespy.nodes import GaussianARD, GaussianGamma, SumMultiply, Gamma

from bayespy.inference import VB
import numpy as np

class BayesianRegression(object):
    """Bayesian linear regression."""
    def __init__(self, n_feature, prior_mean=0, prior_precision=1e-6, prior_a=10, prior_b=1,
                 freeze_weights=False):
        super().__init__()
        self.n_feature = n_feature

        if np.shape(prior_mean) != (n_feature,):
            prior_mean = prior_mean * np.ones(n_feature)
        if np.shape(prior_precision) != (n_feature, n_feature):
            prior_precision = prior_precision * np.ones(n_feature)
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.freeze_weights = freeze_weights
        self._init_weights()
        # print("Intialize regression")
        # self.print()

    def _init_weights(self):
        self.weights = GaussianARD(self.prior_mean, self.prior_precision,
            shape=(self.n_feature,))

    def fit(self, X, y):
        if self.freeze_weights:
            return
        self._init_weights()
        # self.cost,
        # self.myopic_voc(action, state),
        # self.vpi_action(action, state),
        # self.vpi(state),
        # self.expected_term_reward(state)

        self.tau = Gamma(self.prior_a, self.prior_b)
        F = SumMultiply('i,i', self.weights, X)
        y_obs = GaussianARD(F, self.tau)
        y_obs.observe(y)

        Q = VB(y_obs, self.weights)
        Q.update(repeat=10, tol=1e-4, verbose=False)

    def predict(self, x, return_var=False):
        if return_var:
            y = SumMultiply('i,i', self.weights, x)
            y_hat, var, *_ = y.get_moments()
            return y_hat, var
        else:
            w = self.weights.get_moments()[0]
            return x @ w

    def sample(self, x):
        w = self.weights.random()
        return x @ w

    def print(self, diagonal=True):
        mean, m2 = self.weights.get_moments()[:2]
        var = m2 - mean ** 2
        if diagonal:
            var = np.diagonal(var)
        bar = '_' * 40
        print(f'{bar}\n{mean.round(3)}\n{var.round(3)}\n{bar}')


from tqdm import tqdm
from collections import deque

class BayesianQLearner(Policy,):
    """Learns a linear Q function by Bayesian regression."""
    def __init__(self, n_feature, prior_params, sample_weights=True, mem_size=1000, **kwargs):
        super().__init__(**kwargs)
        # assert n_feature == 5
        # if prior_params is None:
            # prior_params = np.r_[np.zeros(4) , 1, np.ones(4)*0.1, 100, 10,1]
        k = n_feature
        self.Q = BayesianRegression(k,
            prior_mean = prior_params[:k],
            prior_precision = prior_params[k:k*2],
            prior_a = prior_params[k*2],
            prior_b = prior_params[k*2+1]
        )
        self.data = {
            'phis': deque(maxlen=mem_size),
            'qs': deque(maxlen=mem_size),
        }
        self.sample_weights = sample_weights
        self.weight_log = []
        self.eval_mode = False


    def attach(self, agent):
        super().attach(agent)

    def eval(self):
        self.eval_mode = True
        self.Q.freeze_weights = True

    def finish_episode(self, trace):
        # if (self.i_episode % 10) == 0:
        #     self.Q.print()

        returns = np.flip(np.cumsum(np.flip(trace['rewards'], 0)), 0)
        self.data['qs'].extend(returns)
        assert len(self.data['qs']) == len(self.data['phis'])
        self.Q.fit(np.stack(self.data['phis']), np.array(self.data['qs']))
        self.weight_log.append(self.Q.weights.get_moments()[0])

    def act(self, state):
        actions = list(self.env.actions(state))
        phi = list(map(self.env.action_features, actions))
        if self.eval_mode:
            v = self.Q.predict(phi, return_var=False)
        elif self.sample_weights:
            v = self.Q.sample(phi)
        else:
            q, var = self.Q.predict(phi, return_var=True)
            v = q + np.sqrt(var) * np.random.randn()

        v += 1e-5 * np.random.rand(len(v))
        i = np.argmax(v)
        self.data['phis'].append(phi[i])
        return actions[i]


