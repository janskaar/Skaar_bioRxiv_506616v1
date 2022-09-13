import matplotlib.pyplot as plt
import numpy as np
import os, pickle
import gpytorch
import torch
import nflows
from deep_lmc_natural_gradients import DeepLMC
from sklearn.neighbors import KernelDensity

TICK_FONT_SIZE = 7
TITLE_FONT_SIZE = 7
LEGEND_FONT_SIZE = 5
LABEL_FONT_SIZE = 7

torch.manual_seed(12345)
np.random.seed(123456)

path = os.path.join('../simulation_scripts')
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

keep_ind = 9

stds = []
expectations = []
for i in range(800):
    d = os.path.join(posterior_dir, f'{i:04d}')
    sample = np.load(os.path.join(d, 'proposals.npy'))
    num_chains = sample.shape[1]
    acceptance = np.load(os.path.join(d, 'acceptance.npy'))
    sample  = np.concatenate([sample[:,i,:][acceptance[:,i]] for i in range(num_chains)])
    stds.append(sample.std(0))
    expectations.append(sample.mean(0))
    if i == keep_ind:
        rate_sample = sample
stds = np.array(stds)
expectations = np.array(expectations)
errors = np.abs(expectations - test_x.numpy())

lfp_stds = []
lfp_expectations = []
for i in range(800):
    d = os.path.join(lfp_posterior_dir, f'{i:04d}')
    sample = np.load(os.path.join(d, 'proposals.npy'))
    num_chains = sample.shape[1]
    acceptance = np.load(os.path.join(d, 'acceptance.npy'))
    sample  = np.concatenate([sample[:,i,:][acceptance[:,i]] for i in range(num_chains)])
    lfp_stds.append(sample.std(0))
    lfp_expectations.append(sample.mean(0))
    if i == keep_ind:
        lfp_sample = sample
lfp_stds = np.array(lfp_stds)
lfp_expectations = np.array(lfp_expectations)
lfp_errors = np.abs(lfp_expectations - test_x.numpy())

index = keep_ind
x0 = test_x[index:index+1]
y0_hist = torch.tensor(test_y_hist[index:index+1])
y0_lfp = torch.tensor(test_y_lfp[index:index+1])

fs = np.linspace(0,500,129)
with gpytorch.settings.num_likelihood_samples(10), torch.no_grad():
    hist_mean, hist_variance = hist_model.batch_mean_variance(x0, likelihood=True)
    hist_std = np.sqrt(hist_variance.numpy())
    hist_mean = hist_mean.numpy() + Y_hist_loc

fs = np.linspace(0,500,129)
with gpytorch.settings.num_likelihood_samples(10), torch.no_grad():
    lfp_mean, lfp_variance = lfp_model.batch_mean_variance(x0, likelihood=True)
    lfp_std = np.sqrt(lfp_variance.numpy())
    lfp_mean = lfp_mean.numpy() + Y_lfp_loc

fig, ax = plt.subplot_mosaic('''
                             ICDE.RLMN
                             .FGH..OPQ
                             ..JK...ST
                             ...X....Y
                             .........
                             WWWW.ZZZZ
                             WWWW.ZZZZ
                             .........
                             UUUU.VVVV
                             UUUU.VVVV
                             .........
                             AAAA.BBBB
                             AAAA.BBBB
                             ''',
                             gridspec_kw={'width_ratios': [1., 1., 1., 1., 0.2, 1., 1., 1., 1.]})

fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.set_size_inches([7,6])
for key in ax:
    ax[key].tick_params(which='both', labelsize=TICK_FONT_SIZE)

x = np.linspace(0,1,50, dtype=np.float32)
xx, yy = np.meshgrid(*[x]*2)
xgrid = np.dstack([xx, yy]).reshape((-1,2))

##############################
# Plot marginal distributions
##############################
marginal_params = [0, 1, 3, 6]
marginal_indices = [(0,1), (0,3), (0,6), (1,3), (1,6), (3,6)]
xticklabels = [('1.0', '3.5'), ('4.5', '8.0'), ('1.0', '8.0'), ('15', '30'), ('0.1', '3.9'), ('0.0', '5.0'), ('25', '100'), ('10', '25')]
labels = ['$\\eta$', '$g$', '$\\tau_s$', '$\\tau_m$', 'delay', '$t_{ref}$', '$Q_s$', '$\\theta$', '$C_{mem}$', '$V_{reset}$']

ax_names = ['C', 'D', 'E', 'G', 'H', 'K']
kdes = []
for i, inds in enumerate(marginal_indices):
    j, k = inds[0], inds[1]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(rate_sample[:,[j, k]])
    gridprob = np.exp(kde.score_samples(xgrid))
    ax[ax_names[i]].pcolormesh(xx, yy, gridprob.reshape((50,50)).T, vmax=gridprob.max(), vmin=0., shading='auto')
    ax[ax_names[i]].scatter(x0[0,k], x0[0,j], c='red', s=2.)
    ax[ax_names[i]].set_xlim(0,1)
    ax[ax_names[i]].set_ylim(0,1)
    ax[ax_names[i]].set_xticks([])
    #ax[ax_names[i]].set_xticklabels([xticklabels[k][0], xticklabels[k][1]])
    ax[ax_names[i]].set_yticks([])
    #ax[ax_names[i]].set_yticklabels([xticklabels[j][0], xticklabels[j][1]])
    #ax[ax_names[i]].set_ylabel(labels[j], fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=-1.5)
    #ax[ax_names[i]].set_xlabel(labels[k], fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=-1.5)

ax_names = ['I', 'F', 'J', 'X']
for i in range(4):
    ind = marginal_params[i]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(rate_sample[:,ind:ind+1])
    gridprob = np.exp(kde.score_samples(x[:,None]))
    ax[ax_names[i]].plot(x, gridprob, lw=0.8, c='black')
    ax[ax_names[i]].set_xlabel(labels[ind], fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=-1.5)
    ax[ax_names[i]].set_xlim(0,1)
    ax[ax_names[i]].set_xticks([0., 1.])
    ax[ax_names[i]].set_xticklabels([xticklabels[ind][0], xticklabels[ind][1]])
    ax[ax_names[i]].set_yticks([])
    ax[ax_names[i]].vlines(x0[0,ind], 0, gridprob.max(), color='red')

ax_names = ['L', 'M', 'N', 'P', 'Q', 'T']
kdes = []
for i, inds in enumerate(marginal_indices):
    j, k = inds[0], inds[1]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(lfp_sample[:,[j, k]])
    gridprob = np.exp(kde.score_samples(xgrid))
    ax[ax_names[i]].pcolormesh(xx, yy, gridprob.reshape((50,50)).T, vmax=gridprob.max(), vmin=0., shading='auto')
    ax[ax_names[i]].scatter(x0[0,k], x0[0,j], c='red', s=2.)
    ax[ax_names[i]].set_xlim(0,1)
    ax[ax_names[i]].set_ylim(0,1)
    ax[ax_names[i]].set_xticks([])
    # ax[ax_names[i]].set_xticklabels([xticklabels[k][0], xticklabels[k][1]])
    ax[ax_names[i]].set_yticks([])
    # ax[ax_names[i]].set_yticklabels([xticklabels[j][0], xticklabels[j][1]])
    ax[ax_names[i]].set_ylabel(labels[j], fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=-1.5)
    ax[ax_names[i]].set_xlabel(labels[k], fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=-1.5)

ax_names = ['R', 'O', 'S', 'Y']
for i in range(4):
    ind = marginal_params[i]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(lfp_sample[:,ind:ind+1])
    gridprob = np.exp(kde.score_samples(x[:,None]))
    ax[ax_names[i]].plot(x, gridprob, lw=0.8, c='black')
    ax[ax_names[i]].set_xlabel(labels[ind], fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=-1.5)
    ax[ax_names[i]].set_xlim(0,1)
    ax[ax_names[i]].set_xticks([0,1])
    ax[ax_names[i]].set_xticklabels([xticklabels[ind][0], xticklabels[ind][1]])
    ax[ax_names[i]].set_yticks([])
    ax[ax_names[i]].vlines(x0[0,ind], 0, gridprob.max(), color='red')

# for i in ['F', 'I', 'J', 'O', 'R', 'S']:
#     ax[i].axis('off')

for i in ['D', 'E', 'H', 'M', 'N', 'Q', 'L', 'P', 'T']:
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_xlabel('')
    ax[i].set_ylabel('')

#for i in ['C', 'D', 'E', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'Q', 'T', 'I', 'F', 'J', 'X']:
    #ax[i].tick_params(which='both', pad=-1)
    #ax[i].set_aspect(1.)

###################################
# Plot simulations from posteriors
###################################

# rate
num_plot_samples = 50

hist_posterior_psds = np.log10(np.load('posterior_simulations_rate.npy')).reshape((-1,50,2*129))
hist_posterior_labels = np.load('posterior_simulations_rate_labels.npy')
plot_samples = hist_posterior_psds[index,:num_plot_samples,:129]
for i in range(num_plot_samples):
    ax['U'].plot(fs, plot_samples[i], c='black', lw=0.7, alpha=0.8, label='simulation' if i == 0 else None)

upper = hist_mean[0][:129] + 2 * hist_std[0][:129]
lower = hist_mean[0][:129] - 2 * hist_std[0][:129]

ax['U'].fill_between(fs, lower, upper, color='C1', alpha=0.3)
ax['U'].plot(fs, hist_mean[0][:129], 'C1', linestyle='--', label='metamodel')

ax['U'].set_xlabel('f (Hz)', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=-1    )
ax['U'].set_ylabel('log $P_{\\nu_E} (f)$', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=-1.5)
ax['U'].legend(fontsize=LEGEND_FONT_SIZE)

# lfp
lfp_posterior_psds = np.log10(np.load('posterior_simulations_lfp.npy')).reshape((-1,50,6*129))
lfp_posterior_labels = np.load('posterior_simulations_lfp_labels.npy')
plot_samples = lfp_posterior_psds[index,:num_plot_samples,:129]

for i in range(num_plot_samples):
    ax['V'].plot(fs, plot_samples[i], c='black', lw=0.7, alpha=0.8)

upper = lfp_mean[0][:129] + 2 * lfp_std[0][:129]
lower = lfp_mean[0][:129] - 2 * lfp_std[0][:129]

ax['V'].fill_between(fs, lower, upper, color='C1', alpha=0.3)
ax['V'].plot(fs, lfp_mean[0][:129], 'C1', linestyle='--')

ax['V'].set_yticks([0, -8])

ax['V'].set_ylabel('log $P_\phi (\mathbf{r}, f)$', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=0.)
ax['V'].set_xlabel('f (Hz)', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=-1)

#####################################
# Plot marginal standard deviations
#####################################
ax['W'].plot(np.arange(10), stds.mean(0), 'o', color='black', markerfacecolor='none', label='pop. activity')
ax['W'].plot(np.arange(10), lfp_stds.mean(0), 'x', color='grey', label='lfp')
ax['W'].set_xticks(np.arange(10))
ax['W'].set_xticklabels(labels)
ax['W'].tick_params(which='both', labelsize=TICK_FONT_SIZE)
ax['W'].set_xticks(np.arange(10))
ax['W'].set_xticklabels(labels)
ax['W'].tick_params(which='both', labelsize=TICK_FONT_SIZE)
ax['W'].set_ylim(0., 0.3)
ax['W'].set_ylabel('marginal std.', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=0.5)
ax['W'].legend(fontsize=LEGEND_FONT_SIZE)

ax['Z'].plot(np.arange(10), errors.mean(0), 'o', color='black', markerfacecolor='none')
ax['Z'].plot(np.arange(10), lfp_errors.mean(0), 'x', color='grey')
ax['Z'].set_xticks(np.arange(10))
ax['Z'].set_xticklabels(labels)
ax['Z'].tick_params(which='both', labelsize=TICK_FONT_SIZE)
ax['Z'].set_xticks(np.arange(10))
ax['Z'].set_xticklabels(labels)
ax['Z'].tick_params(which='both', labelsize=TICK_FONT_SIZE)
ax['Z'].set_ylim(0., 0.3)
ax['Z'].set_ylabel('abs. error (stds)', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=0.5)

###################################
# Plot posterior simulation errors
###################################

lfp_hist_posterior_psds = np.log10(np.load('posterior_simulations_lfp_rate.npy')).reshape((-1,50,2*129))
rate_lfp_posterior_psds = np.log10(np.load('posterior_simulations_rate_lfp.npy')).reshape((-1,50,6*129))

with gpytorch.settings.num_likelihood_samples(10), torch.no_grad():
    hist_mean, hist_variance = hist_model.batch_mean_variance(test_x[:100], likelihood=True)
    hist_std = np.sqrt(hist_variance.numpy())
    hist_mean = hist_mean.numpy() + Y_hist_loc

with gpytorch.settings.num_likelihood_samples(10), torch.no_grad():
    lfp_mean, lfp_variance = lfp_model.batch_mean_variance(test_x[:100], likelihood=True, batch_size=10)
    lfp_std = np.sqrt(lfp_variance.numpy())
    lfp_mean = lfp_mean.numpy() + Y_lfp_loc

posterior_rate_errors = (hist_posterior_psds - hist_mean[:,None,:]) / hist_std[:,None,:]
posterior_lfp_errors = (lfp_posterior_psds - lfp_mean[:,None,:]) / lfp_std[:,None,:]

posterior_lfp_rate_errors = (lfp_hist_posterior_psds - hist_mean[:,None,:]) / hist_std[:,None,:]
posterior_rate_lfp_errors = (rate_lfp_posterior_psds - lfp_mean[:,None,:]) / lfp_std[:,None,:]

ax['A'].hist(np.abs(posterior_rate_errors).max(2).flatten(), bins=np.linspace(1.8,12,51), density=True, color='black', histtype='step', label='simulations pop. activity posterior')
ax['A'].hist(np.abs(posterior_lfp_rate_errors).max(2).flatten(), bins=np.linspace(1.8,12,51), density=True, color='grey', histtype='step', alpha=0.8, label='simulations LFP posterior')
ax['B'].hist(np.abs(posterior_lfp_errors).max(2).flatten(), bins=np.linspace(1.8,12,51), density=True, color='grey', histtype='step')
ax['B'].hist(np.abs(posterior_rate_lfp_errors).max(2).flatten(), bins=np.linspace(1.8,12,51), density=True, color='black', histtype='step', alpha=0.8)
ax['A'].set_ylim(0, 1)
ax['B'].set_ylim(0, 1)
ax['A'].set_xlabel('max error (stds)', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=0.)
ax['A'].set_ylabel('density', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=0.)
ax['B'].set_xlabel('max error (stds)', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=0.)
ax['A'].legend(fontsize=LEGEND_FONT_SIZE)
#ax['B'].legend(fontsize=LEGEND_FONT_SIZE)

labels = ['A', 'B', 'C', 'D']
pos = [(0.08, 0.89), (0.08, 0.59), (0.08, 0.42), (0.08, 0.23)]
for i in range(4):
    tax = fig.add_axes([*pos[i], 0.05, 0.05])
    tax.axis('off')
    tax.text(0, 0, labels[i], fontdict={'fontsize': 12, 'rotation': 0})


fig.savefig('results_fig6.pdf', bbox_inches='tight')
plt.show()
