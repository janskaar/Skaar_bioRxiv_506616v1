import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from sklearn.cross_decomposition import PLSRegression
from HCPLSR import HCPLSR
import os
import gpytorch
import torch
from deep_lmc_natural_gradients import DeepLMC

torch.manual_seed(12345)
np.random.seed(123456)

TICK_FONT_SIZE = 7
TITLE_FONT_SIZE = 7
LABEL_FONT_SIZE = 7

sims_path = os.path.join('../training_scripts')
model_paths_num_samples = [os.path.join(sims_path, f'deep_lmc_10p_hist_nosync_{i}', 'deep_lmc_5.pth') for i in [100, 200, 400, 800, 1600]]
model_paths_num_samples.append(os.path.join(sims_path, f'deep_lmc_10p_hist_nosync', 'deep_lmc_5.pth'))

path = os.path.join('../simulation_scripts')
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

num_samples = [100, 200, 400, 800, 1600, 3000]
test_errors_num_samples = []
test_variances_num_samples = []
for model_path in model_paths_num_samples:
    model = DeepLMC(test_x.shape, num_latent_gps=14, num_hidden_dgp_dims=10, num_tasks=258)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    with gpytorch.settings.num_likelihood_samples(10), torch.no_grad():
        gp_test_mean, gp_test_variance = model.batch_mean_variance(test_x, likelihood=True)
        gp_test_mean = gp_test_mean.numpy()
        gp_test_variance = gp_test_variance.numpy()
    test_errors_num_samples.append(np.abs(gp_test_mean - test_y).max(axis=1).mean())
    test_variances_num_samples.append(gp_test_variance.mean())

hcpls_test_errors_num_samples = []
X_exp = np.concatenate([X] + [np.expand_dims(X[:,i]*X[:,j], 1) for i in range(10) for j in range(i, 10)], axis=1)
global_components = [1,2]
local_components = [20, 22, 24, 26, 28, 30, 32, 34]
num_clusters = [2, 4, 6]
for i, n in enumerate(num_samples):
    minloss = 1e10
    for j, _ in enumerate(global_components):
        for k, _ in enumerate(local_components):
            for l, _ in enumerate(num_clusters):
                try:
                    hcpls = HCPLSR(X[:n], Y[:n],
                                   global_components=global_components[j],
                                   local_components=local_components[k],
                                   n_clusters=num_clusters[l],
                                   fuzzy_m=2.0)
                    preds = hcpls.predict(X[3000:3800])
                    error = np.abs(preds - Y_test).max(1).mean()
                    if error < minloss:
                        minloss = error
                        use_hcpls = hcpls
                except:
                    continue

    hcpls_test_preds = use_hcpls.predict(X[3000:3800])
    hcpls_test_errors_num_samples.append(np.abs(hcpls_test_preds - Y_test).max(1).mean())

pls_test_errors_num_samples = []
for i, n in enumerate(num_samples):
    pls = PLSRegression(n_components=30, scale=True)
    pls.fit(X_exp[:n], Y_train[:n])
    pls_test_preds = pls.predict(X_exp[3000:3800])
    pls_test_errors_num_samples.append(np.abs(pls_test_preds - Y_test).max(1).mean())

pls_test_errors_num_samples = [min((pls_test_errors_num_samples[i], hcpls_test_errors_num_samples[i])) for i, _ in enumerate(num_samples)]

fig, ax = plt.subplots(ncols=2)
fig.set_size_inches([7,1.5])
fig.subplots_adjust(wspace=0.4, hspace=0.4, bottom=0.3)

ax[0].plot(num_samples, pls_test_errors_num_samples, 'o-', markersize=2., label='HCPLS')
ax[0].plot(num_samples, test_errors_num_samples, 'o-', markersize=2., label='GP')
ax[1].plot(num_samples, np.sqrt(test_variances_num_samples), 'o-', markersize=2., c='C1')
ax[0].legend(fontsize=TICK_FONT_SIZE)

ax[0].set_ylabel('mean max error (abs.)', fontdict={'fontsize': LABEL_FONT_SIZE})
ax[1].set_ylabel('mean std', fontdict={'fontsize': LABEL_FONT_SIZE})
ax[0].set_xlabel('training samples', fontdict={'fontsize': LABEL_FONT_SIZE})
ax[1].set_xlabel('training samples', fontdict={'fontsize': LABEL_FONT_SIZE})

for a in ax:
    a.tick_params(which='both', axis='both', labelsize=TICK_FONT_SIZE)
ax[0].set_yticks([0.2, 0.3, 0.4, 0.5])
ax[1].set_yticks([0.061, 0.065, 0.07])

labels = ['A', 'B']
pos = [(0.08, 0.95), (0.52, 0.95)]
for i in range(2):
    tax = fig.add_axes([*pos[i], 0.05, 0.05])
    tax.axis('off')
    tax.text(0, 0, labels[i], fontdict={'fontsize': 10, 'rotation': 0})

fig.savefig('results_fig5.pdf', bbox_inches='tight')
plt.show()
