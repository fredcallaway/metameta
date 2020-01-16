import random
import numpy as np
import pandas as pd

# Prevents annoying crash
import matplotlib; matplotlib.use('agg')

from skopt import gp_minimize, expected_minimum
from skopt.space import Real,Space
from joblib import Parallel, delayed, load, dump

from mouselab import *
from distributions import *
from bayes_q import *
from gymrats.core import Agent
from exact import solve


cost_range = [0,0.2]
branch_range = [2,4]
height_range = [2,4]
reward_alpha = 1

N_FEATURE = 5
if N_FEATURE == 5:
    DIMENSIONS = [Real(1, 20), Real(0,1), Real(0,1), Real(0,1), Real(0,2)]
else:
    assert N_FEATURE == 4
    DIMENSIONS = [Real(-5, 5), Real(1,20), Real(0,2), Real(0,2)]

def sample_env():
    cost = np.random.uniform(*cost_range)
    branch = np.random.randint(*branch_range)
    height = np.random.randint(*height_range)

    p = np.random.uniform(.1, .9)
    x = (1-p) / p
    R = Categorical([-1, x], [(1-p), p])
    # R = Categorical([-1, 0, 1],np.random.dirichlet(np.ones(3)*reward_alpha))

    env = MouselabEnv.new_erdos_renyi(30, reward=R, cost=cost)
    if N_FEATURE == 4:
        env.simple_features = True
    return env

def create_training_df(env, prior_params, n_episode):
    policy = BayesianQLearner(N_FEATURE, prior_params=prior_params)
    agent = Agent(env, policy)
    return pd.DataFrame(agent.run_many(n_episode, pbar=False))

def make_prior_params(x):
    if len(x) == N_FEATURE:  # no learning
        return np.r_[x[:N_FEATURE], np.ones(N_FEATURE) * 1e10, 1, 1]
    else:
        assert len(x) == N_FEATURE * 2 + 2
        return x

def training_return(x, env=None, seed=None, n_episode=20):
    if env is None:
        env = sample_env()
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    df = create_training_df(env, make_prior_params(x), n_episode)
    baseline = env.expected_term_reward(env.init)
    # baseline = create_training_df(env, make_prior_params([0, 1, 1, 0]), n_episode).return_.mean()
    return np.mean(df.return_.values) - baseline

def make_objective(single, N, parallel=True, **kwargs):
    def objective(x):
        if parallel:
            jobs = (delayed(single)(x, seed=np.random.randint(100000000), **kwargs) for _ in range(N))
            returns = Parallel(-1, backend='multiprocessing')(jobs)
        else:
            returns = [single(x, **kwargs) for _ in range(N)]
        y = np.mean(returns)
        print(objective.counter, np.round(y, 3), np.round(x,2))
        objective.counter += 1
        return y
    objective.counter = 0
    return objective


def train_agent(kind, n_iter=200, n_env=100, n_episode=25, **kwargs):
    dimensions = list(DIMENSIONS)
    if kind == 'meta':
        dimensions.extend([Real(0.1,100)]*N_FEATURE + [Real(0.1,20), Real(0.1,10)])

    objective = make_objective(training_return, N=n_env, n_episode=n_episode)
    return gp_minimize(lambda x: - objective(x),
                       dimensions=dimensions, n_calls=n_iter,
                        **kwargs)


# %% ====================  ====================

def expand_x(x):
    return [x[0], 1, x[1], 1]

def optimize_global(N=5000, parallel=True):
    objective = make_objective(training_return, N=N, n_episode=1, parallel=parallel)

    dimensions = [Real(-1., 1.), Real(0., 2.)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gp_minimize(lambda x: -objective(expand_x(x)), dimensions=dimensions, n_calls=100, n_random_starts=25)
    return res


global_res = optimize_global(5000)

objective = make_objective(training_return, N=5000, n_episode=1)
objective(expand_x(global_res.x))
objective(expand_x(x_greedy))
objective(expand_x(expected_minimum(global_res)))
global_x = global_res.x
dump('pickle/global_x_4')

# %% ====================  ====================
meta_res = train_agent('meta', n_episode=40, n_env=1000, n_random_starts=50)
print("GLOBAL")
global_res = train_agent('global', n_episode=40, n_env=1000, n_random_starts=50)

objective = make_objective(training_return, N=1000, n_episode=40)

objective(global_res.x)
objective(expected_minimum(global_res)[0])
objective(meta_res.x)
objective(expected_minimum(meta_res)[0])

# meta_x = expected_minimum(meta_res)[0]


# %% ====================  ====================
x = global_res.x
# envs = [sample_env() for _ in range(1000)]
vs = Parallel(-1)(delayed(training_return)(x, env, 1, 1) for env in envs)
np.std(vs)

# %% ====================  ====================
def training_returns(env, prior, seed):
    np.random.seed(seed)
    random.seed(seed)
    return list(create_training_df(env, prior, 40).return_)

def get_training_returns(envs, x):
    prior = make_prior_params(x)
    jobs = [delayed(training_returns)(env, prior, seed)
            for seed, env in enumerate(envs)]
    return Parallel(-1)(jobs)

envs = [sample_env() for _ in range(1000)]
meta_curves = get_training_returns(envs, meta_res.x)
global_curves = get_training_returns(envs, global_res.x)

dump(meta_curves, 'pickle/meta_curves4')
dump(global_curves, 'pickle/global_curves4')

# %% ==================== Find environments for which global solution is suboptimal  ====================
x_greedy = [0, 1, 1, 0]
global_x =

def select_envs(n):
    envs = []
    while len(envs) < n:
        env = sample_env()
        df = create_training_df(env, make_prior_params(global_x), 10)
        if np.std(df.n_steps) != 0 and np.std(df.return_) != 0:
            print('#', end='')
            envs.append(env)
    return envs

envs = select_envs(90)

def mean_sem(x):
    a, b = np.mean(x), np.std(x)/np.sqrt(len(x))
    print('{:.3f} ± {:.3f}'.format(a, b))
    return a, b

def eval_fixed_env(x, env, N, parallel=False):
    if parallel:
        jobs = (delayed(training_return)(x, env=env, seed=np.random.randint(1e8), n_episode=1) for i in range(N))
        return Parallel(-1, backend='multiprocessing')(jobs)
    else:
        return [training_return(x, env=env, seed=np.random.randint(1e8), n_episode=1) for i in range(N)]

import warnings
def optimize_env(env, N=1000, parallel=False, verbose=False):

    def loss(x):
        y = -np.mean(eval_fixed_env(expand_x(x), env, N, parallel=parallel))
        if verbose:
            print(np.round(x, 2), '->', round(y, 2))
        return y

    dimensions = [Real(-1., 1.), Real(0., 2.)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gp_minimize(loss, dimensions=dimensions, n_calls=100, n_random_starts=25)
    return res

# %% ====================  ====================
from skopt import expected_minimum
res = optimize_env(envs[0], 5000, True, True)
print('Observed:', end=' ')
mean_sem(eval_fixed_env(expand_x(res.x), envs[0], 5000, True))
print('Expected:', end=' ')
mean_sem(eval_fixed_env(expand_x(expected_minimum(res)[0]), envs[0], 5000, True))
print('Global:  ', end=' ')
mean_sem(eval_fixed_env(expand_x([0, 1]), envs[0], 5000, True))


# %% ====================  ====================
opt_results = Parallel(-1, backend='multiprocessing')(delayed(optimize_env)(env) for env in envs)
local_xs = [res.x for res in opt_results]
local_vals = [eval_fixed_env(x, env, 5000, parallel=True)
              for x, env in zip(local_xs, envs)]
global_vals = [eval_fixed_env(global_x, env, 5000, parallel=True)
              for env in envs]

# %% ====================  ====================
def run(env):
    global_vals = eval_fixed_env(global_x, env, 5000, True)
    mean_sem(global_vals)
    res = optimize_env(env, parallel=True, verbose=False)
    local_vals = eval_fixed_env(expand_x(res.x), env, 5000, True)
    mean_sem(local_vals)
    # mean_sem(global_vals)
    return (res, global_vals, local_vals)

results = []
for i, env in enumerate(envs):
    print(f"-------- {i} --------")
    results.append(run(env))


# %% ====================  ====================
from scipy.stats import ttest_ind

def process(i):
    return {
        'p': ttest_ind(local_vals[i], global_vals[i]).pvalue,
        'local': np.mean(local_vals[i]),
        'global_': np.mean(global_vals[i])
    }

df = pd.DataFrame(map(process, range(len(envs))))
local_better = df.local > df.global_
significant = df.p < .001
diff = (df.local - df.global_)
rel_diff = diff / abs(df.global_)

print("Local: ", end=''); mean_sem(df.local)
print("Global: ", end=''); mean_sem(df.global_)
print("Rel Diff: ", rel_diff.mean())

print("Significant better: ", (local_better & significant).mean())
print("Significant worse: ", (~local_better & significant).mean())


# %% ====================  ====================
test_results = []
for (i, env) in enumerate(envs):
    print(f'----- {i} -----')
    test_results.append(test_env(env, parallel=True))


# %% ====================  ====================
jobs = [delayed(test_env)(env) for env in envs]
test_results = Parallel(-1)(jobs)




