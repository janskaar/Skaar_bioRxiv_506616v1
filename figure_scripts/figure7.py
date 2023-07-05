import matplotlib.pyplot as plt
import numpy as np
import os, pickle

plt.rcParams["font.size"] = 7

posterior_dir = os.path.join('../train_posterior_distributions/mcmc_posterior_samples_dgp')
lfp_posterior_dir = os.path.join('../train_posterior_distributions/mcmc_posterior_samples_lfp_2ch_dgp')

covariances = []
correlations = []
expectations = []
for i in range(100):
    d = os.path.join(posterior_dir, f'{i:04d}')
    sample = np.load(os.path.join(d, 'proposals.npy'))
    num_chains = sample.shape[1]
    acceptance = np.load(os.path.join(d, 'acceptance.npy'))
    sample  = np.concatenate([sample[:,i,:][acceptance[:,i]] for i in range(num_chains)])
    mean = sample.mean(0)
    sample -= mean
    cov = sample.T.dot(sample) / (len(sample) - 1)
    stds = np.sqrt(np.diag(cov))
    expectations.append(mean)
    covariances.append(cov)
    corr = cov / stds[None,:]
    corr /= stds[:,None]
    np.fill_diagonal(corr, 0.)
    correlations.append(corr)

covariances = np.array(covariances)
expectations = np.array(expectations)
correlations = np.array(correlations)

lfp_covariances = []
lfp_expectations = []
lfp_correlations = []
for i in range(100):
    d = os.path.join(lfp_posterior_dir, f'{i:04d}')
    sample = np.load(os.path.join(d, 'proposals.npy'))
    num_chains = sample.shape[1]
    acceptance = np.load(os.path.join(d, 'acceptance.npy'))
    sample  = np.concatenate([sample[:,i,:][acceptance[:,i]] for i in range(num_chains)])
    mean = sample.mean(0)
    sample -= mean
    cov = sample.T.dot(sample) / (len(sample) - 1)
    stds = np.sqrt(np.diag(cov))
    corr = cov / stds[None,:]
    corr /= stds[:,None]
    np.fill_diagonal(corr, 0.)

    lfp_expectations.append(mean)
    lfp_covariances.append(cov)
    lfp_correlations.append(corr)

lfp_covariances = np.array(lfp_covariances)
lfp_expectations = np.array(lfp_expectations)
lfp_correlations = np.array(lfp_correlations)

a = correlations.mean(0)
b = lfp_correlations.mean(0)
np.fill_diagonal(a, 0.)
np.fill_diagonal(b, 0.)
vmin, vmax = min((a.min(), b.min())), max((a.max(), b.max()))
vmin, vmax = -max(abs(vmin), abs(vmax)), max(abs(vmin), abs(vmax))

correlations_plot = np.zeros((10, 10))
np.fill_diagonal(correlations_plot, np.nan)
triu = np.triu_indices(10, 1)
correlations_plot[triu] = correlations.mean(0)[triu]
tril = np.tril_indices(10, -1)
correlations_plot[tril] = lfp_correlations.mean(0)[tril]


# %%

cmap = plt.get_cmap('bwr').copy()
cmap.set_bad(color='black', alpha=1.)
labels = ['$\\eta$', '$g$', '$\\tau_s$', '$\\tau_m$', 'delay', '$t_{ref}$', '$Q_s$', '$V_{thr}$', '$C_{mem}$', '$V_{reset}$']
fig, ax = plt.subplots(1)
fig.set_size_inches([7,7])
plot = ax.pcolormesh(correlations_plot[::-1,:], vmin=vmin, vmax=vmax, cmap=cmap )
ax.set_xticks(np.arange(10) + 0.5)
ax.set_yticks(np.arange(10) + 0.5)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels[::-1])
ax.tick_params(which='both')
ax.set_aspect(1.)
ax.set_ylim(0.,10.)
ax.set_xlim(0.,10.)
twax = ax.twinx()
twax.set_aspect(1.)

twax.tick_params(which='both')
twax.xaxis.tick_top()
twax.xaxis.set_visible(True)
twax.set_xticks(np.arange(10) + 0.5)
twax.set_xticklabels(labels)
twax.set_yticks(np.arange(10) + 0.5)
twax.set_yticklabels(labels[::-1])
twax.set_xlim(0., 10.)
twax.set_ylim(0., 10.)

cbar = fig.colorbar(plot)
fig.savefig('figure7.pdf', bbox_inches='tight')
plt.show()
