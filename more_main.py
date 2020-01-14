


# %% ====================  ====================
def make_df(test_results):
    rows = []
    for tr in test_results:
        g, o = tr
        rows.append([*g, *o])
    return pd.DataFrame(rows, columns=['global_mean', 'global_sem', 'local_mean', 'local_sem'])


df = make_df(test_results)
(df.local_mean - df.global_mean).mean()


# %% ====================  ====================
from scipy.stats import ttest_ind_from_stats
def apply_ttest(tr, n=500):
    (g_mean, g_sem), (o_mean, o_sem) = tr
    g_std = g_sem * np.sqrt(n)
    o_std = o_sem * np.sqrt(n)
    return ttest_ind_from_stats(g_mean, g_std, n, o_mean, o_std, n, equal_var=False).pvalue


ps = np.array(list(map(apply_ttest, test_results)))
significant = ps < .05
print(f'Proportion significant: {np.mean(significant):.2f}')
d = df.iloc[significant]
d.local_mean - d.global_mean

# %% ====================  ====================

good = significant & (df.local_mean > df.global_mean)
good_envs = np.array(envs)[good]
for env in good_envs:
    # print(env.branch[0], env.height)
    print(env.reward)


# def count_calls(func):
#     def wrapped(*args, **kwargs):
#         y = func(*args, **kwargs)
#         wrapped.i += 1
#         return y
#     wrapped.i = 0
#     return wrapped
#
# @count_calls
# def loss(x):
#     return -np.mean(eval_fixed_env(x, env, 500))

# dimensions = [Real(1, 20), Real(0,1), Real(0,1), Real(0,1), Real(0,2)]
# res = gp_minimize(loss, dimensions=dimensions, n_calls=100)

# In[32]:


plt.plot(np.mean(np.stack(no_learn_trs), 0), label='Fixed')
plt.plot(np.mean(np.stack(trs), 0), label='Learning')
plt.legend();


# In[30]:


env = sample_env()
def job(env):
    return create_training_df(env, make_prior_params(x), 20).return_

res = Parallel(-1)(delayed(job)() for _ in range(48))


# In[46]:


plt.plot(np.mean(np.stack(res), 0))


# In[ ]:


global_res = train_agent('global', n_iter=100)
# meta_res = train_agent('meta', n_iter=100, )


# In[18]:


global_res.x


# In[21]:


from skopt import expected_minimum

def choose_x(res):
    objective = make_objective(training_return, N=1000)
    x_exp, f_exp_pred = expected_minimum(res)
    f_exp = objective(x_exp)
    x_emp, f_emp_pred = res.x, res.fun
    f_emp = objective(x_emp)
    print(f'Expected: {np.round(x_exp, 2)}  ->  {f_exp:.3f}  ({-f_exp_pred:.3f})')
    print(f'Empirical: {np.round(x_emp, 2)}  ->  {f_emp:.3f}  ({-f_emp_pred:.3f})')
    return x_exp if f_exp > f_emp else x_emp

x_meta = choose_x(meta_res)


# In[53]:


# from skopt import expected_minimum
# x_global, f_global = expected_minimum(global_res)
# print(np.round(x_global, 3))

# prm = make_prior_params(x_global)
# f_emp = make_objective(training_return, N=1000)(prm)
# print(f'Expected: {-f_global:.3f}')
# print(f'Empirical: {f_emp:.3f}')

# prm_100 = make_prior_params(x_global)
# prm_100[5:10] = 100
# f_100 = make_objective(training_return, N=1000)(prm_100)

# print(f'100: {f_100}')

# meta_res = train_agent('meta', n_iter=200, x0=[prm_100.tolist()], y0=[f_100])


# In[54]:


from tqdm import tqdm

def train(env, prior_params, n_episode, seed):
    np.random.seed(seed)
    random.seed(seed)
    df = create_training_df(env, prior_params, n_episode)
    return df.return_


def train_many(envs, x, n_episode=20):
    prior_params = make_prior_params(x)
    jobs = (delayed(train)(env, prior_params, n_episode, np.random.randint(10000000)) for env in envs)
    return Parallel(-1)((jobs))

x_baseline = list(np.r_[np.zeros(5), np.ones(5)*.01, 1,.1])
num_replications = 1000
envs = [sample_env() for _ in range(num_replications)]
# %time global_returns = train_many(envs, x_global)
get_ipython().run_line_magic('time', 'meta_returns = train_many(envs, x_meta)')
get_ipython().run_line_magic('time', 'baseline_returns = train_many(envs, x_baseline)')


# In[58]:


def plot_returns(X, **kwargs):
    plt.errorbar(x=np.arange(X.shape[1]),y=np.mean(X,axis=0),
             yerr=np.std(X,axis=0)/np.sqrt(X.shape[0]), **kwargs)

# plot_returns(np.stack(global_returns), label='Global')
plot_returns(np.stack(baseline_returns), label='Baseline')
plot_returns(np.stack(meta_returns), label='Meta')
plt.xlabel("Episode")
plt.ylabel("Return")


# In[98]:



m = pd.melt(pd.DataFrame(np.stack(meta_returns)), var_name='Episode', value_name='Return')
sns.lineplot(m.Episode, m.Return)

n = pd.melt(pd.DataFrame(np.stack(baseline_returns)), var_name='Episode', value_name='Return')
sns.lineplot(n.Episode, n.Return, color=(0.6, 0.6, 0.6))


plt.legend(['Metalearned Prior', 'Uninformative Prior'], loc='lower right')
plt.savefig('figs/results.pdf',  pad_inches=0.1, bbox_inches='tight')


# In[137]:


plt.errorbar(x=np.arange(global_returns.shape[1]),y=np.mean(global_returns,axis=0),
             yerr=np.std(global_returns,axis=0)/np.sqrt(global_returns.shape[0]),label='mean return of learned policy')
plt.errorbar(x=np.arange(meta_returns.shape[1]),y=np.mean(meta_returns,axis=0),
             yerr=np.std(meta_returns,axis=0)/np.sqrt(meta_returns.shape[0]),label='mean return of meta-learned policy')
#plt.errorbar(np.mean(returns_meta,axis=0),'.-',label='mean return of meta-learned policy')
#plt.plot(pd.DataFrame(np.mean(returns,axis=0)).rolling(10).mean(),'.-',label='mean return of learned policy')
#plt.axhline(np.mean(optimal_return),color='black',linestyle='dashed',label='return of optimal policy')
plt.legend(frameon=False)

plt.show()

#plt.show()
#df.n_steps.rolling(1).mean().plot()
#plt.show()


# In[145]:


np.savetxt('returns_notmeta.txt',returns)
np.savetxt('returns_meta.txt',returns_meta)


# In[149]:


x = np.loadtxt('returns_notmeta.txt')
y = np.loadtxt('returns_meta.txt')
plt.plot(np.mean(x,axis=0))
plt.plot(np.mean(y,axis=0))
