import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import os
import gpytorch
import torch
from sklearn.cross_decomposition import PLSRegression
from HCPLSR import HCPLSR
from deep_lmc_natural_gradients import DeepLMC

TICK_FONT_SIZE = 7
TITLE_FONT_SIZE = 7
LABEL_FONT_SIZE = 7

torch.manual_seed(12345)
np.random.seed(123456)

path = os.path.join('../simulation_scripts/')
psds = np.load(os.path.join(path, '10p_hist_psd_nosync.npy')).astype(np.float32)
psds = psds.reshape((-1, 258))
labels = np.load(os.path.join(path, '10p_labels_nosync.npy')).astype(np.float32)

minlabels = labels.min(axis=0)
difflabels = labels.max(axis=0) - labels.min(axis=0)
labels -= minlabels
labels /= difflabels

X = labels.copy()
Y = np.log10(psds.copy())
X_train = X[:3000]
Y_train = Y[:3000]
Y_loc = Y_train.mean(axis=0)
Yn_train = Y_train - Y_loc

X_test = X[3000:3800]
Y_test = Y[3000:3800]
Yn_test = Y_test - Y_loc

test_y = Yn_test
test_x = torch.tensor(X_test)

model = DeepLMC(test_x.shape, num_latent_gps=14, num_hidden_dgp_dims=10, num_tasks=258)
state_dict = torch.load(f'../training_scripts/deep_lmc_10p_hist_nosync/deep_lmc_4.pth')
model.load_state_dict(state_dict)
model.eval()
with gpytorch.settings.num_likelihood_samples(10), torch.no_grad():
    gp_test_mean, gp_test_variance = model.batch_mean_variance(test_x, likelihood=True)
    gp_test_mean = gp_test_mean.numpy()
    gp_test_variance = gp_test_variance.numpy()
gp_test_errors = np.abs(gp_test_mean - test_y)
gp_test_errors_mean = gp_test_errors.mean(axis=-1)
gp_test_errors_max = gp_test_errors.max(axis=-1)

gp_test_errors_max_sortargs = gp_test_errors_max.argsort()


hcpls = HCPLSR(X[:3000], Y[:3000], global_components=2, local_components=34, n_clusters=6, fuzzy_m=2.0)
pls_test_preds = hcpls.predict(X[3000:3800])

# X_exp = np.concatenate([X] + [np.expand_dims(X[:,i]*X[:,j], 1) for i in range(10) for j in range(i, 10)], axis=1)
# pls = PLSRegression(n_components=30, scale=True)
# pls.fit()
# pls_test_preds = pls.predict(X_exp[3000:3800])

pls_test_errors = np.abs(pls_test_preds - Y[3000:3800])
print(pls_test_errors.mean())
pls_test_errors_mean = pls_test_errors.mean(axis=-1)
pls_test_errors_max = pls_test_errors.max(axis=-1)
pls_test_errors_max_sortargs = pls_test_errors_max.argsort()
fs = np.linspace(0,500,129)

fig, axs = plt.subplot_mosaic('''CCAA
                                 CCBB''')
fig.subplots_adjust(wspace=0.4, hspace=0., bottom=0.2)
fig.set_size_inches([7,1.5])

gp_test_mean += Y_loc

AB_ind = gp_test_errors_max_sortargs[620]

upper = gp_test_mean[AB_ind][:129] + 2 * np.sqrt(gp_test_variance[AB_ind][:129])
lower = gp_test_mean[AB_ind][:129] - 2 * np.sqrt(gp_test_variance[AB_ind][:129])

axs['A'].plot(fs, test_y[AB_ind, :129] + Y_loc[:129], color='black', lw=1., label='simulation')
axs['A'].fill_between(fs, lower, upper, alpha=0.2, color='C1')
axs['A'].plot(fs, gp_test_mean[AB_ind, :129], color='C1', lw=0.8, label='GP')
axs['A'].plot(fs, pls_test_preds[AB_ind, :129], color='C0', lw=0.8, label='HCPLS')
axs['A'].legend(fontsize=TICK_FONT_SIZE, loc='upper right')

upper = gp_test_mean[AB_ind][129:] + 2 * np.sqrt(gp_test_variance[AB_ind][129:])
lower = gp_test_mean[AB_ind][129:] - 2 * np.sqrt(gp_test_variance[AB_ind][129:])

axs['B'].plot(fs, test_y[AB_ind, 129:] + Y_loc[129:], color='black', lw=1.)
axs['B'].fill_between(fs, lower, upper, alpha=0.2, color='C1')
axs['B'].plot(fs, gp_test_mean[AB_ind, 129:], color='C1', lw=0.8)
axs['B'].plot(fs, pls_test_preds[AB_ind, 129:], color='C0', lw=0.8)
axs['A'].get_shared_x_axes().join(axs['A'], axs['B'])
axs['A'].get_shared_y_axes().join(axs['A'], axs['B'])
axs['A'].set_xticklabels([])

num_test = len(gp_test_errors_max)
axs['C'].bar(np.arange(0,num_test,10), pls_test_errors_max[gp_test_errors_max_sortargs[::10]], width=10., label='HCPLS')
axs['C'].bar(560, pls_test_errors_max[AB_ind], width=10., color='red')
axs['C'].bar(np.arange(0,num_test,10), gp_test_errors_max[gp_test_errors_max_sortargs[::10]], width=10., label='GP', alpha=0.6)
axs['C'].legend(fontsize=TICK_FONT_SIZE)

axs['A'].set_ylabel('log $P_{\\nu_E} (f)$', fontdict={'fontsize': LABEL_FONT_SIZE})
axs['B'].set_ylabel('log $P_{\\nu_I} (f)$', fontdict={'fontsize': LABEL_FONT_SIZE})

for i in ['A', 'B', 'C']:
    axs[i].tick_params(which='both', axis='both', labelsize=TICK_FONT_SIZE)

axs['C'].set_xlabel('index', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=0.)
axs['C'].set_ylabel('max error (abs.)', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=0.)

for i in ['B']:
    axs[i].set_xlabel('f (Hz)', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=0.)

labels = ['A', 'B']
pos = [(0.08, 0.9), (0.51, 0.9)]
for i in range(2):
    tax = fig.add_axes([*pos[i], 0.05, 0.05])
    tax.axis('off')
    tax.text(0, 0, labels[i], fontdict={'fontsize': 12, 'rotation': 0})


fig.savefig('results_fig4.pdf', bbox_inches='tight')
plt.show()

1
