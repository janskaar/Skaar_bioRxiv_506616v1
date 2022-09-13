import matplotlib.pyplot as plt
import numpy as np
import os, pickle
import gpytorch
import torch
import nflows
from deep_lmc_natural_gradients import DeepLMC
from sklearn.neighbors import KernelDensity

TICK_FONT_SIZE = 5
TITLE_FONT_SIZE = 5
LABEL_FONT_SIZE = 5

torch.manual_seed(12345)
np.random.seed(123456)

path = os.path.join('../simulation_scripts/')
hist_psds = np.load(os.path.join(path, '10p_hist_psd_nosync.npy')).astype(np.float32)
hist_psds = hist_psds.reshape((-1, 258))
lfp_psds = np.load(os.path.join(path, '10p_lfp_psd_nosync.npy')).astype(np.float32)
lfp_psds = lfp_psds.reshape((-1, 6*129))
labels = np.load(os.path.join(path, '10p_labels_nosync.npy')).astype(np.float32)

minlabels = labels.min(axis=0)
difflabels = labels.max(axis=0) - labels.min(axis=0)
labels -= minlabels
labels /= difflabels

X = labels.copy()
X_train = X[:3000]
X_test = X[3000:3800]

Y_hist = np.log10(hist_psds.copy())
Y_hist_train = Y_hist[:3000]
Y_hist_loc = Y_hist_train.mean(axis=0)

Y_lfp = np.log10(lfp_psds.copy())
Y_lfp_train = Y_lfp[:3000]
Y_lfp_loc = Y_lfp_train.mean(axis=0)

Y_hist_test = Y_hist[3000:3800]
Yn_hist_test = Y_hist_test - Y_hist_loc

Y_lfp_test = Y_lfp[3000:3800]
Yn_lfp_test = Y_lfp_test - Y_lfp_loc

test_y_hist = Yn_hist_test
test_y_lfp = Yn_lfp_test
test_x = torch.tensor(X_test)

hist_model = DeepLMC(X.shape, num_latent_gps=14, num_hidden_dgp_dims=10, num_tasks=258)
state_dict = torch.load('../training_scripts/deep_lmc_10p_hist_nosync/deep_lmc_4.pth')
hist_model.load_state_dict(state_dict)
hist_model.eval();

lfp_model = DeepLMC(X.shape, num_latent_gps=14, num_hidden_dgp_dims=10, num_tasks=6*129)
state_dict = torch.load('../training_scripts/deep_lmc_10p_lfp_nosync/deep_lmc_4.pth')
lfp_model.load_state_dict(state_dict)
lfp_model.eval();

posterior_dir = os.path.join('../posterior_mcmc_scripts/run_mcmc_posteriors_rate')
lfp_posterior_dir = os.path.join('../posterior_mcmc_scripts/run_mcmc_posteriors_lfp')

covariances = []
correlations = []
expectations = []
for i in range(800):
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
for i in range(800):
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
ax.tick_params(which='both', labelsize=TICK_FONT_SIZE)
ax.set_aspect(1.)
ax.set_ylim(0.,10.)
ax.set_xlim(0.,10.)
twax = ax.twinx()
twax.set_aspect(1.)

twax.tick_params(which='both', labelsize=TICK_FONT_SIZE)
twax.xaxis.tick_top()
twax.xaxis.set_visible(True)
twax.set_xticks(np.arange(10) + 0.5)
twax.set_xticklabels(labels)
twax.set_yticks(np.arange(10) + 0.5)
twax.set_yticklabels(labels[::-1])
twax.set_xlim(0., 10.)
twax.set_ylim(0., 10.)

cbar = fig.colorbar(plot)
cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
fig.savefig('results_fig8.pdf', bbox_inches='tight')
plt.show()
