import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os
import gpytorch
import torch
from deep_lmc_natural_gradients import DeepLMC

TICK_FONT_SIZE = 7
TITLE_FONT_SIZE = 7
LABEL_FONT_SIZE = 7

torch.manual_seed(12345678)
np.random.seed(1234567)

path = os.path.join('../simulation_scripts/')
psds = np.load(os.path.join(path, '10p_hist_psd_nosync.npy')).astype(np.float32)
psds = psds.reshape((-1, 258))
labels = np.load(os.path.join(path, '10p_labels_nosync.npy')).astype(np.float32)

lfp_psds = np.load(os.path.join(path, '10p_lfp_psd_nosync.npy')).astype(np.float32)
lfp_psds = lfp_psds.reshape((-1, 6*129))

psds_x10 = np.load(os.path.join(path, '10p_hist_psd_x10.npy')).astype(np.float32)
labels_x10 = np.load(os.path.join(path, '10p_labels_x10.npy')).astype(np.float32)
psds_x10 = np.log10(psds_x10)
psds_x10 = psds_x10.reshape((-1, 258))

lfp_psds_x10 = np.load(os.path.join(path, '10p_lfp_psd_x10.npy')).astype(np.float32)
lfp_psds_x10 = np.log10(lfp_psds_x10)
lfp_psds_x10 = lfp_psds_x10.reshape((-1, 6*129))

minlabels = labels.min(axis=0)
difflabels = labels.max(axis=0) - labels.min(axis=0)
labels -= minlabels
labels /= difflabels

X = labels.copy()
Y = np.log10(psds.copy())
X_train = X[:3000]
Y_train = Y[:3000]
Y_loc = Y_train.mean(axis=0)

Y_lfp = np.log10(lfp_psds.copy())
Y_lfp_train = Y_lfp[:3000]
Y_lfp_loc = Y_lfp_train.mean(axis=0)

X_test = X[3000:3800]
Y_test = Y[3000:3800]
Yn_test = Y_test - Y_loc

Y_lfp_test = Y_lfp[3000:3800]
Yn_lfp_test = Y_lfp_test - Y_lfp_loc

test_y = Yn_test
test_x = torch.tensor(X_test)
test_y_lfp = Yn_lfp_test

labels_x10 -= minlabels
labels_x10 /= difflabels

x_x10 = torch.tensor(labels_x10)
y_x10 = psds_x10 - Y_loc
y_lfp_x10 = lfp_psds_x10 - Y_lfp_loc

######################################
### Load rate model
######################################
num_inducing1 = 128
num_inducing2 = 128
model = DeepLMC(test_x.shape,
                num_latent_gps=14,
                num_hidden_dgp_dims=10,
                num_tasks=258,
                num_inducing1=num_inducing1,
                num_inducing2=num_inducing2
                )

state_dict = torch.load('../training_scripts/deep_lmc_10p_hist_nosync/deep_lmc_4.pth')
model.load_state_dict(state_dict)
model.eval()

#################################################
### Predictions on test set and errors
#################################################
with gpytorch.settings.num_likelihood_samples(10), torch.no_grad():
    test_mean, test_variance = model.batch_mean_variance(test_x, likelihood=True)
    test_std = np.sqrt(test_variance.numpy())
    test_mean = test_mean.numpy()

test_errors = (test_mean - test_y)
test_errors_abs = np.abs(test_errors)
test_errors_abs_norm = np.abs(test_errors / test_std)
sortargs_max = test_errors_abs_norm.max(axis=1).argsort()

with gpytorch.settings.num_likelihood_samples(10), torch.no_grad():
    output_x10 = model.data_independent_dist(x_x10[:100], likelihood=True)
    full_mean_x10 = output_x10.mean
    cov_x10 = output_x10.covariance_matrix
    mean_x10 = output_x10.loc.numpy().mean(axis=0)
    variance_x10 = torch.mean(output_x10.mean**2 + output_x10.variance, axis=0) - output_x10.mean.mean(axis=0)**2
    variance_x10 = variance_x10.numpy()
    std_x10 = np.sqrt(variance_x10)

errors_x10 = np.abs(mean_x10 - y_x10[:100]) / std_x10
sortargs_max_x10 = errors_x10.max(axis=1).argsort()

y_x10 += Y_loc
#######################################
### Figure out which examples to show
### based on errors from rate model
#######################################
error_10 = sorted(test_errors_abs_norm.max(axis=1))[int(len(test_errors_abs) * 0.1)]
AB_ind = np.abs(errors_x10.max(axis=1) - error_10).argmin()
error_90 = sorted(test_errors_abs_norm.max(axis=1))[int(len(test_errors_abs) * 0.90)]
CD_ind = np.abs(errors_x10.max(axis=1) - error_90).argmin()

######################################
### Load lfp model
######################################
lfp_model = DeepLMC(test_x.shape,
                num_latent_gps=14,
                num_hidden_dgp_dims=10,
                num_tasks=129*6,
                num_inducing1=num_inducing1,
                num_inducing2=num_inducing2
                )

state_dict = torch.load('../training_scripts/deep_lmc_10p_lfp_nosync/deep_lmc_4.pth')
lfp_model.load_state_dict(state_dict)
lfp_model.eval()

#######################################
### LFP predictions on test set
#######################################
with gpytorch.settings.num_likelihood_samples(10), torch.no_grad():
    test_mean_lfp, test_variance_lfp = lfp_model.batch_mean_variance(test_x, likelihood=True, batch_size=20)
    test_std_lfp = np.sqrt(test_variance_lfp.numpy())
    test_mean_lfp = test_mean_lfp.numpy()

test_errors_lfp = test_mean_lfp - test_y_lfp
test_errors_lfp_abs = np.abs(test_errors_lfp)
test_errors_lfp_abs_norm = np.abs(test_errors_lfp) / test_std_lfp
sortargs_lfp_max = test_errors_lfp_abs_norm.max(axis=1).argsort()

with gpytorch.settings.num_likelihood_samples(10), torch.no_grad():
    output_lfp_x10 = lfp_model.data_independent_dist(x_x10[[AB_ind, CD_ind]], likelihood=True)
    full_mean_lfp_x10 = output_lfp_x10.mean
    cov_lfp_x10 = output_lfp_x10.covariance_matrix
    mean_lfp_x10 = output_lfp_x10.loc.numpy().mean(axis=0)
    variance_lfp_x10 = torch.mean(output_lfp_x10.mean**2 + output_lfp_x10.variance, axis=0) - output_lfp_x10.mean.mean(axis=0)**2
    variance_lfp_x10 = variance_lfp_x10.numpy()
    std_lfp_x10 = np.sqrt(variance_lfp_x10)

y_lfp_x10 += Y_lfp_loc

########################################
### Create figure
########################################
fs = np.linspace(0,500,129)
fig, axs = plt.subplot_mosaic('''
                              AAAAAA..VVVVVV
                              AAAAAA..VVVVVV
                              AAAAAA..VVVVVV
                              AAAAAA..VVVVVV
                              AAAAAA..VVVVVV
                              ..............
                              CCCCCC..XXXXXX
                              CCCCCC..XXXXXX
                              CCCCCC..XXXXXX
                              CCCCCC..XXXXXX
                              CCCCCC..XXXXXX
                              ..............
                              ..............
                              ..............
                              ....FFFFFF....
                              ....FFFFFF....
                              ....FFFFFF....
                              ....FFFFFF....
                              ....FFFFFF....
                              ....FFFFFF....
                              ..............
                              ..............
                              ..............
                              GGHHIIJJKKLLMM
                              GGHHIIJJKKLLMM
                              ..............
                              OOPPQQRRSSTTUU
                              OOPPQQRRSSTTUU
                              ''',
                              gridspec_kw={})

fig.subplots_adjust(wspace=0.3, hspace=0.)

fig.set_size_inches([7, 5])

###########################################
###                AB                   ###
###########################################
upper = mean_x10[AB_ind][:129] + Y_loc[:129] + 2 * std_x10[AB_ind][:129]
lower = mean_x10[AB_ind][:129] + Y_loc[:129] - 2 * std_x10[AB_ind][:129]
axs['A'].fill_between(fs, lower, upper, alpha=0.2, color='C1')
sample_mvn = gpytorch.distributions.MultivariateNormal(full_mean_x10[:,AB_ind,:129], cov_x10[:,AB_ind,:129,:129])
samples = sample_mvn.sample().detach().numpy()
samples += Y_loc[:129]

for i in range(10):
    axs['A'].plot(fs, samples[i], color='C1', lw=0.6, label='metamodel' if i == 0 else None)
    axs['A'].plot(fs, y_x10[AB_ind + i * 100,:129], color='black', lw=0.6, label='simulation' if i == 0 else None)

axs['A'].legend(fontsize=TICK_FONT_SIZE, loc='upper right')

# axs['B'].fill_between(fs, lower, upper, alpha=0.2, color='C1')
# sample_mvn = gpytorch.distributions.MultivariateNormal(full_mean_x10[:,AB_ind,129:], cov_x10[:,AB_ind,129:,129:])
# samples = sample_mvn.rsample().detach().numpy()
# samples += Y_loc[129:]
#
# for i in range(10):
#     axs['B'].plot(fs, samples[i], color='C1', lw=0.6)
#     axs['B'].plot(fs, y_x10[AB_ind + i * 100,129:], color='black', lw=0.6)

###########################################
###                CD                   ###
###########################################
upper = mean_x10[CD_ind][:129] + Y_loc[:129] + 2 * np.sqrt(variance_x10[CD_ind][:129])
lower = mean_x10[CD_ind][:129] + Y_loc[:129] - 2 * np.sqrt(variance_x10[CD_ind][:129])
axs['C'].fill_between(fs, lower, upper, alpha=0.2, color='C1')
sample_mvn = gpytorch.distributions.MultivariateNormal(full_mean_x10[:,CD_ind,:129], cov_x10[:,CD_ind,:129,:129])
samples = sample_mvn.rsample().detach().numpy()
samples += Y_loc[:129]

for i in range(10):
    axs['C'].plot(fs, samples[i], color='C1', lw=0.6)
    axs['C'].plot(fs, y_x10[CD_ind + i * 100,:129], color='black', lw=0.6)

# upper = mean_x10[CD_ind][129:] + Y_loc[129:] + 2 * np.sqrt(variance_x10[CD_ind][129:])
# lower = mean_x10[CD_ind][129:] + Y_loc[129:] - 2 * np.sqrt(variance_x10[CD_ind][129:])
# axs['D'].fill_between(fs, lower, upper, alpha=0.2, color='C1')
# sample_mvn = gpytorch.distributions.MultivariateNormal(full_mean_x10[:,CD_ind,129:], cov_x10[:,CD_ind,129:,129:])
# samples = sample_mvn.rsample().detach().numpy()
# samples += Y_loc[129:]

# for i in range(10):
#     axs['D'].plot(fs, samples[i], color='C1', lw=0.6)
#     axs['D'].plot(fs, y_x10[CD_ind + i * 100,129:], color='C0', lw=0.6)

###########################################
###                VW                   ###
###########################################
upper = mean_lfp_x10[0][:129] + Y_lfp_loc[:129] + 2 * np.sqrt(variance_lfp_x10[0][:129])
lower = mean_lfp_x10[0][:129] + Y_lfp_loc[:129] - 2 * np.sqrt(variance_lfp_x10[0][:129])
axs['V'].fill_between(fs, lower, upper, alpha=0.2, color='C1')
sample_mvn = gpytorch.distributions.MultivariateNormal(full_mean_lfp_x10[:,0,:129], cov_lfp_x10[:,0,:129,:129])
samples = sample_mvn.sample().detach().numpy()
samples += Y_lfp_loc[:129]

for i in range(10):
    axs['V'].plot(fs, samples[i], color='C1', lw=0.6, label='metamodel' if i == 0 else None)
    axs['V'].plot(fs, y_lfp_x10[AB_ind + i * 100,:129], color='black', lw=0.6, label='simulation' if i == 0 else None)

#axs['V'].legend(fontsize=TICK_FONT_SIZE, loc='upper right')

# upper = mean_lfp_x10[0][4*129:5*129] + Y_lfp_loc[4*129:5*129] + 2 * np.sqrt(variance_lfp_x10[0][4*129:5*129])
# lower = mean_lfp_x10[0][4*129:5*129] + Y_lfp_loc[4*129:5*129] - 2 * np.sqrt(variance_lfp_x10[0][4*129:5*129])
#
# axs['W'].fill_between(fs, lower, upper, alpha=0.2, color='C1')
# sample_mvn = gpytorch.distributions.MultivariateNormal(full_mean_lfp_x10[:,0,:129], cov_lfp_x10[:,0,4*129:5*129,4*129:5*129])
# samples = sample_mvn.rsample().detach().numpy()
# samples += Y_lfp_loc[4*129:5*129]
#
# for i in range(10):
#     axs['W'].plot(fs, samples[i], color='C1', lw=0.6)
#     axs['W'].plot(fs, y_lfp_x10[AB_ind + i * 100,4*129:5*129], color='C0', lw=0.6)

###########################################
###                XY                   ###
###########################################
upper = mean_lfp_x10[1][:129] + Y_lfp_loc[:129] + 2 * np.sqrt(variance_lfp_x10[1][:129])
lower = mean_lfp_x10[1][:129] + Y_lfp_loc[:129] - 2 * np.sqrt(variance_lfp_x10[1][:129])
axs['X'].fill_between(fs, lower, upper, alpha=0.2, color='C1')
sample_mvn = gpytorch.distributions.MultivariateNormal(full_mean_lfp_x10[:,1,:129], cov_lfp_x10[:,1,:129,:129])
samples = sample_mvn.sample().detach().numpy()
samples += Y_lfp_loc[:129]

for i in range(10):
    axs['X'].plot(fs, samples[i], color='C1', lw=0.6)
    axs['X'].plot(fs, y_lfp_x10[CD_ind + i * 100,:129], color='black', lw=0.6)

# upper = mean_lfp_x10[1][4*129:5*129] + Y_lfp_loc[4*129:5*129] + 2 * np.sqrt(variance_lfp_x10[1][4*129:5*129])
# lower = mean_lfp_x10[1][4*129:5*129] + Y_lfp_loc[4*129:5*129] - 2 * np.sqrt(variance_lfp_x10[1][4*129:5*129])
# axs['Y'].fill_between(fs, lower, upper, alpha=0.2, color='C1')
# sample_mvn = gpytorch.distributions.MultivariateNormal(full_mean_lfp_x10[:,1,4*129:5*129], cov_lfp_x10[:,1,4*129:5*129,4*129:5*129])
# samples = sample_mvn.sample().detach().numpy()
# samples += Y_lfp_loc[4*129:5*129]
#
# for i in range(10):
#     axs['Y'].plot(fs, samples[i], color='C1', lw=0.6)
#     axs['Y'].plot(fs, y_lfp_x10[CD_ind + i * 100,4*129:5*129], color='C0', lw=0.6)

###########################################
###                E                    ###
###########################################
# fig, axs = plt.subplot_mosaic('''
#                               E
#                               ''')
# x = np.linspace(-2.5, 2.5, 2)
# for j, i in enumerate(frequency_indices[1:2]):
#     e = test_errors[:,i].copy()
#     e -= e.mean()
#     e /= e.std()
#     e.sort()
#     quants = (np.arange(1, 801) - 0.5) / (800 + 1 - 2 * 0.5) # https://en.wikipedia.org/wiki/Normal_probability_plot
#     norm_ppf = norm.ppf(quants, loc=0, scale=1.)
#     e_sparse = np.concatenate([e[:30], e[30:-30:10], e[-30:]])
#     ppf_sparse = np.concatenate([norm_ppf[:30], norm_ppf[30:-30:10], norm_ppf[-30:]])
#     axs['E'].plot(ppf_sparse, -j * 0.2 + ppf_sparse, 'black', lw=1.)
#     axs['E'].scatter(ppf_sparse, - j * 0.2 + e_sparse, s=1., label=f'{int(fs[i])} Hz')

#axs['E'].set_ylim(-1.5, 0.5)
#axs['E'].legend(fontsize=TICK_FONT_SIZE, bbox_to_anchor=(-0.02, 0.9))

###########################################
###                F                    ###
###########################################
axs['F'].plot(test_errors_abs_norm[sortargs_max].max(axis=1), label='rate model')
axs['F'].plot(test_errors_lfp_abs_norm[sortargs_lfp_max].max(axis=1), label='LFP model')
#axs['F'].set_yticks([0., 0.5, 1., 1.5])
axs['F'].set_xticks([0, 800])
axs['F'].tick_params(which='both', pad=-1)
axs['F'].set_xlabel('index', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=-1)
axs['F'].set_ylabel('max error (stds)', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=-0.3)
#axs['F'].set_ylim(0,1.6)
axs['F'].legend(fontsize=TICK_FONT_SIZE)

##########################################
###               G-M                  ###
##########################################
frequency_indices = [0, 20, 40, 60, 80, 100, 120]
bins = np.linspace(-0.3, 0.3, 50)
for i, a in enumerate(['G', 'H', 'I', 'J', 'K', 'L', 'M']):
    axs[a].hist(test_errors[:,frequency_indices[i]], bins=bins, density=True)
    axs[a].set_ylim(0., 9.)
    axs[a].set_title(f'f = {int(fs[frequency_indices[i]])} Hz', fontsize=LABEL_FONT_SIZE)
    axs[a].set_xticklabels([])
    if i != 0:
        axs[a].set_yticklabels([])

for i, a in enumerate(['O', 'P', 'Q', 'R', 'S', 'T', 'U']):
    axs[a].hist(test_errors_lfp[:,frequency_indices[i]], bins=bins, density=True)
    axs[a].set_ylim(0., 9.)
    axs[a].set_xlabel('error', fontdict={'fontsize': LABEL_FONT_SIZE})
    if i != 0:
        axs[a].set_yticklabels([])

#####################
# axis stuff
#####################
axs['A'].set_xticks([])
#axs['B'].set_xticks([])
#axs['C'].set_xticks([])
axs['V'].set_xticks([])
#axs['W'].set_xticks([])
#axs['X'].set_xticks([])
#axs['E'].set_xticks([])
#axs['E'].set_yticks([])

axs['A'].set_ylim(.5, 6.0)
#axs['B'].set_ylim(.5, 6.0)
axs['C'].set_ylim(.5, 6.0)
#axs['D'].set_ylim(.5, 6.0)
axs['V'].set_ylim(-11, 1)
#axs['W'].set_ylim(-11, 1)
axs['X'].set_ylim(-11, 1)
#axs['Y'].set_ylim(-11, 1)

for ax in axs.items():
    ax[1].tick_params(which='both', axis='both', labelsize=TICK_FONT_SIZE)

for i in ['A','C']:
    axs[i].set_ylabel('log $P_{\\nu_X} (f)$', fontdict={'fontsize': LABEL_FONT_SIZE})
for i in ['V', 'X']:
    axs[i].set_ylabel('log $P_{\\phi} (f)$', fontdict={'fontsize': LABEL_FONT_SIZE})

axs['C'].set_xlabel('$f\ (s^{-1})$',  fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=-1)
axs['X'].set_xlabel('$f\ (s^{-1})$',  fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=-1)

labels = ['A', 'B', 'C', 'D']
pos = [(0.08, 0.9), (0.52, 0.9), (0.32, 0.5), (0.08, 0.26)]
for i in range(4):
    tax = fig.add_axes([*pos[i], 0.05, 0.05])
    tax.axis('off')
    tax.text(0, 0, labels[i], fontdict={'fontsize': 12, 'rotation': 0})


fig.savefig('results_fig3.pdf', bbox_inches='tight')
plt.show()
