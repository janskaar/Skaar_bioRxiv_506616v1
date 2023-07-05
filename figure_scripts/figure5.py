import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.signal import welch
from sklearn.neighbors import KernelDensity
import os, pickle, h5py

plt.rcParams["font.size"] = 7

n_train = 8000
index = 6 # index to show plots for


##################################################
## Load main data set
##################################################

with h5py.File('lfp_kernel_delta.h5', 'r') as f:
    ex_kernel_delta = f['ex'][()]
    in_kernel_delta = f['in'][()]

# Alpha function for computing LFPs
def isyn_fn(J, tau):
    t = np.arange(0, 100, 0.1)
    return - J * t/tau * np.exp(1 - t/tau) * 1e-3

n_train = 8000

sim_path = os.path.join("../simulation_scripts/brunel_simulations_10p")
dirs = sorted(os.listdir(os.path.join(sim_path, "nest_output")))

params = []
hists = []
lfp_psds = []
hist_psds = []
# loop through directories containing individual simulations
for i, d in enumerate(dirs):
    with open(os.path.join(sim_path, "parameters", d + ".pkl"), "rb") as f:
        param = pickle.load(f)
        params.append(np.array([param["eta"], param["g"], param["tauSyn"], param["tauMem"], param["delay"], param["t_ref"], param["cSyn"], param["theta"], param["CMem"], param["V_reset"]]))

    with h5py.File(os.path.join(sim_path, "nest_output", d, "LFP_firing_rate.h5"), "r") as f:
        ex_hist = f["ex_hist"][()]
        in_hist = f["in_hist"][()]

    fs, epsd = welch(ex_hist)
    fs, ipsd = welch(in_hist)
    hist_psds.append(np.concatenate((epsd, ipsd)))

    ex_kernel = []
    in_kernel = []
    ex_isyn = isyn_fn(param['J'], param['tauSyn'])
    in_isyn = isyn_fn(-param['J'] * param['g'], param['tauSyn'])
    for j in range(6):
        ex_kernel.append(np.convolve(ex_isyn, ex_kernel_delta[j]))
        in_kernel.append(np.convolve(in_isyn, in_kernel_delta[j]))
    ex_kernel = np.array(ex_kernel).reshape((6,200,10)).mean(axis=-1)
    in_kernel = np.array(in_kernel).reshape((6,200,10)).mean(axis=-1)

    lfp = []
    for j in range(6):
        lfp.append(np.convolve(ex_hist, ex_kernel[j], mode='valid') + np.convolve(in_kernel[j], in_hist, mode='valid'))
    lfp = np.array(lfp)
    lfp -= lfp.mean(1, keepdims=True)
    fs, psd = welch(lfp)
    lfp_psds.append(psd.flatten())
    hists.append(np.stack((ex_hist, in_hist)))

print("FINISHED LOADING")

lfp_psds = np.array(lfp_psds)
hist_psds = np.array(hist_psds)
hists = np.array(hists)
params = np.array(params)

# remove simulations in which activity is strongly synchronous
args = ((hists[:,0,:] > 800).sum(axis=1) > 150) & ((hists[:,0,:] < 20).sum(axis=1) > 500)

labels = params[~args].copy()
hist_psds = hist_psds[~args]
lfp_psds = lfp_psds[~args]

minlabels = np.array([1.0, 4.5, 1., 15., 0.1, 0.0, 25., 15., 100., 0.])
difflabels = np.array([2.5, 3.5, 7., 15., 3.9, 5., 75., 10., 200., 10.])
labels -= minlabels
labels /= difflabels

X = labels.copy().astype(np.float32)
Y = np.log10(hist_psds.copy().astype(np.float32))
Ytrain_mean = Y[:n_train].mean(0, keepdims=True)
#Yn = Y - Ytrain_mean

#train_x = X[:n_train]
#train_y = Yn[:n_train]

test_x = X[n_train:]
test_y = Y[n_train:]


# LFPs
Y_lfp = np.log10(lfp_psds)
Ytrain_lfp_mean = Y_lfp[:n_train].mean(0, keepdims=True)

test_y_lfp = Y_lfp[n_train:]


######################################################################
## Load posterior distribution and simulations from posterior
######################################################################

# Posterior distributions
posterior_dir = os.path.join('../train_posterior_distributions/mcmc_posterior_samples_dgp')
lfp_posterior_dir = os.path.join('../train_posterior_distributions/mcmc_posterior_samples_lfp_2ch_dgp')

hist_stds = []
hist_expectations = []
for i in range(100):
    d = os.path.join(posterior_dir, f'{i:04d}')
    sample = np.load(os.path.join(d, 'proposals.npy'))
    num_chains = sample.shape[1]
    acceptance = np.load(os.path.join(d, 'acceptance.npy'))
    sample  = np.concatenate([sample[:,i,:][acceptance[:,i]] for i in range(num_chains)])
    hist_stds.append(sample.std(0))
    hist_expectations.append(sample.mean(0))
    if i == index:
        hist_sample = sample
hist_stds = np.array(hist_stds)
hist_expectations = np.array(hist_expectations)
hist_errors = np.abs(hist_expectations - test_x[:100])

lfp_stds = []
lfp_expectations = []
for i in range(100):
    d = os.path.join(lfp_posterior_dir, f'{i:04d}')
    sample = np.load(os.path.join(d, 'proposals.npy'))
    num_chains = sample.shape[1]
    acceptance = np.load(os.path.join(d, 'acceptance.npy'))
    sample  = np.concatenate([sample[:,i,:][acceptance[:,i]] for i in range(num_chains)])
    lfp_stds.append(sample.std(0))
    lfp_expectations.append(sample.mean(0))
    if i == index:
        lfp_sample = sample
lfp_stds = np.array(lfp_stds)
lfp_expectations = np.array(lfp_expectations)
lfp_errors = np.abs(lfp_expectations - test_x[:100])

x0 = test_x[index:index+1]
y0_hist = test_y[index:index+1]
y0_lfp = test_y_lfp[index:index+1]

# load from hist posterior
sim_path = os.path.join("../simulation_scripts/brunel_simulations_dgp_posterior_mcmc")
dirs = sorted(os.listdir(os.path.join(sim_path, "nest_output")))  #  [f"{index*50 + i:04d}" for i in range(50)]
hist_psds = []
hist_posterior_lfp_psds = []
# loop through directories containing individual simulations
for i, d in enumerate(dirs):
    with h5py.File(os.path.join(sim_path, "nest_output", d, "LFP_firing_rate.h5"), "r") as f:
        ex_hist = f["ex_hist"][()]
        in_hist = f["in_hist"][()]

    fs, epsd = welch(ex_hist)
    fs, ipsd = welch(in_hist)
    hist_psds.append(np.concatenate((epsd, ipsd)))

    ex_kernel = []
    in_kernel = []
    ex_isyn = isyn_fn(param['J'], param['tauSyn'])
    in_isyn = isyn_fn(-param['J'] * param['g'], param['tauSyn'])
    for j in range(6):
        ex_kernel.append(np.convolve(ex_isyn, ex_kernel_delta[j]))
        in_kernel.append(np.convolve(in_isyn, in_kernel_delta[j]))
    ex_kernel = np.array(ex_kernel).reshape((6,200,10)).mean(axis=-1)
    in_kernel = np.array(in_kernel).reshape((6,200,10)).mean(axis=-1)

    lfp = []
    for j in range(6):
        lfp.append(np.convolve(ex_hist, ex_kernel[j], mode='valid') + np.convolve(in_kernel[j], in_hist, mode='valid'))
    lfp = np.array(lfp)
    lfp -= lfp.mean(1, keepdims=True)
    fs, psd = welch(lfp)
    hist_posterior_lfp_psds.append(psd.flatten())

hist_posterior_lfp_psds = np.log10(np.array(hist_posterior_lfp_psds)).reshape((100,50,-1))
hist_psds = np.log10(np.array(hist_psds)).reshape((100,50,-1))


# load from lfp posterior
sim_path = os.path.join("../simulation_scripts/brunel_simulations_dgp_posterior_lfp_2ch_mcmc")
dirs = sorted(os.listdir(os.path.join(sim_path, "nest_output")))  #  [f"{index*50 + i:04d}" for i in range(50)]
lfp_psds = []
lfp_posterior_hist_psds = []
# loop through directories containing individual simulations
for i, d in enumerate(dirs):
    with open(os.path.join(sim_path, "parameters", d + ".pkl"), "rb") as f:
        param = pickle.load(f)

    with h5py.File(os.path.join(sim_path, "nest_output", d, "LFP_firing_rate.h5"), "r") as f:
        ex_hist = f["ex_hist"][()]
        in_hist = f["in_hist"][()]

    fs, epsd = welch(ex_hist)
    fs, ipsd = welch(in_hist)
    lfp_posterior_hist_psds.append(np.concatenate((epsd, ipsd)))

    ex_kernel = []
    in_kernel = []
    ex_isyn = isyn_fn(param['J'], param['tauSyn'])
    in_isyn = isyn_fn(-param['J'] * param['g'], param['tauSyn'])
    for j in range(6):
        ex_kernel.append(np.convolve(ex_isyn, ex_kernel_delta[j]))
        in_kernel.append(np.convolve(in_isyn, in_kernel_delta[j]))
    ex_kernel = np.array(ex_kernel).reshape((6,200,10)).mean(axis=-1)
    in_kernel = np.array(in_kernel).reshape((6,200,10)).mean(axis=-1)

    lfp = []
    for j in range(6):
        lfp.append(np.convolve(ex_hist, ex_kernel[j], mode='valid') + np.convolve(in_kernel[j], in_hist, mode='valid'))
    lfp = np.array(lfp)
    lfp -= lfp.mean(1, keepdims=True)
    fs, psd = welch(lfp)
    lfp_psds.append(psd.flatten())

lfp_psds = np.log10(np.array(lfp_psds)).reshape((100,50,-1))
lfp_posterior_hist_psds = np.log10(np.array(lfp_posterior_hist_psds)).reshape((100,50,-1))



posterior_pred_dir = os.path.join("..", "train_conditional_distributions", "compute_predictions_errors")
hist_posterior_hist_preds = np.load(os.path.join(posterior_pred_dir, "dgp_hist_posterior_mean_hist.npy"))
hist_posterior_hist_vars = np.load(os.path.join(posterior_pred_dir, "dgp_hist_posterior_var_hist.npy"))
hist_posterior_lfp_preds = np.load(os.path.join(posterior_pred_dir, "dgp_hist_posterior_mean_lfp.npy"))
hist_posterior_lfp_vars = np.load(os.path.join(posterior_pred_dir, "dgp_hist_posterior_var_lfp.npy"))
lfp_posterior_hist_preds = np.load(os.path.join(posterior_pred_dir, "dgp_lfp_posterior_mean_hist.npy"))
lfp_posterior_hist_vars = np.load(os.path.join(posterior_pred_dir, "dgp_lfp_posterior_var_hist.npy"))
lfp_posterior_lfp_preds = np.load(os.path.join(posterior_pred_dir, "dgp_lfp_posterior_mean_lfp.npy"))
lfp_posterior_lfp_vars = np.load(os.path.join(posterior_pred_dir, "dgp_lfp_posterior_var_lfp.npy"))

hist_test_preds = np.load(os.path.join(posterior_pred_dir, "dgp_test_mean.npy"))
hist_test_vars = np.load(os.path.join(posterior_pred_dir, "dgp_test_var.npy"))
lfp_test_preds = np.load(os.path.join(posterior_pred_dir, "dgp_lfp_test_mean.npy"))
lfp_test_vars = np.load(os.path.join(posterior_pred_dir, "dgp_lfp_test_var.npy"))

##################################################
# Create plots
##################################################

fs = np.linspace(0,500,129)
fig, ax = plt.subplot_mosaic('''
                             ICDE.RLMN
                             .FGH..OPQ
                             ..JK...ST
                             .0.X..1.Y
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
    ax[key].tick_params(which='both')

x = np.linspace(0,1,50, dtype=np.float32)
xx, yy = np.meshgrid(*[x]*2)
xgrid = np.dstack([xx, yy]).reshape((-1,2))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Plot marginal distributions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
marginal_params = [0, 1, 3, 6]
marginal_indices = [(0,1), (0,3), (0,6), (1,3), (1,6), (3,6)]
xticklabels = [('1.0', '3.5'), ('4.5', '8.0'), ('1.0', '8.0'), ('15', '30'), ('0.1', '3.9'), ('0.0', '5.0'), ('25', '100'), ('10', '25')]
labels = ['$\\eta$', '$g$', '$\\tau_s$', '$\\tau_m$', 'delay', '$t_{ref}$', '$Q_s$', '$\\theta$', '$C_{mem}$', '$V_{reset}$']

ax_names = ['C', 'D', 'E', 'G', 'H', 'K']
kdes = []
for i, inds in enumerate(marginal_indices):
    j, k = inds[0], inds[1]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(hist_sample[:,[j, k]])
    gridprob = np.exp(kde.score_samples(xgrid))
    ax[ax_names[i]].pcolormesh(xx, yy, gridprob.reshape((50,50)).T, vmax=gridprob.max(), vmin=0., shading='auto')
    ax[ax_names[i]].scatter(x0[0,k], x0[0,j], c='red', s=2.)
    ax[ax_names[i]].set_xlim(0,1)
    ax[ax_names[i]].set_ylim(0,1)
    ax[ax_names[i]].set_xticks([])
    #ax[ax_names[i]].set_xticklabels([xticklabels[k][0], xticklabels[k][1]])
    ax[ax_names[i]].set_yticks([])
    #ax[ax_names[i]].set_yticklabels([xticklabels[j][0], xticklabels[j][1]])

ax_names = ['I', 'F', 'J', 'X']
for i in range(4):
    ind = marginal_params[i]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(hist_sample[:,ind:ind+1])
    gridprob = np.exp(kde.score_samples(x[:,None]))
    ax[ax_names[i]].plot(x, gridprob, lw=0.8, c='black')
    ax[ax_names[i]].set_xlabel(labels[ind], labelpad=-1.5)
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
    ax[ax_names[i]].set_ylabel(labels[j], labelpad=-1.5)
    ax[ax_names[i]].set_xlabel(labels[k], labelpad=-1.5)

ax_names = ['R', 'O', 'S', 'Y']
for i in range(4):
    ind = marginal_params[i]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(lfp_sample[:,ind:ind+1])
    gridprob = np.exp(kde.score_samples(x[:,None]))
    ax[ax_names[i]].plot(x, gridprob, lw=0.8, c='black')
    ax[ax_names[i]].set_xlabel(labels[ind], labelpad=-1.5)
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


# ground truth labels
ax["0"].axis("off")
ax["1"].axis("off")
ax["0"].scatter([1.], [1.], label="ground truth", color="red", s=3.)
ax["1"].scatter([1.], [1.], label="ground truth", color="red", s=3.)
ax["0"].set_xlim(0,0.1)
ax["1"].set_xlim(0,0.1)
ax["0"].legend()
ax["1"].legend()

ax["C"].set_title("                 Pop. spiking activity")
ax["L"].set_title("                 LFP")

#for i in ['C', 'D', 'E', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'Q', 'T', 'I', 'F', 'J', 'X']:
    #ax[i].tick_params(which='both', pad=-1)
    #ax[i].set_aspect(1.)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Plot simulations from posteriors
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# plot rate
mean = hist_test_preds[index] + Ytrain_mean.squeeze()
std = np.sqrt(hist_test_vars[index])
num_plot_samples = 50
plot_samples = hist_psds[index,:,:129]
for i in range(num_plot_samples):
    ax['U'].plot(fs, plot_samples[i], c='black', lw=0.5, alpha=0.8, label='simulation' if i == 0 else None)

upper = mean[:129] + 2 * std[:129]
lower = mean[:129] - 2 * std[:129]

ax['U'].fill_between(fs, lower, upper, color='C1', alpha=0.3)
ax['U'].plot(fs, mean[:129], 'C1', linestyle='--', label='metamodel')

ax['U'].set_xlabel('f (Hz)', labelpad=-1    )
ax['U'].set_ylabel('log $P_{\\nu_E} (f)$', labelpad=-1.5)
ax['U'].legend()

# lfp
mean = lfp_test_preds[index] + Ytrain_lfp_mean.squeeze()
std = np.sqrt(lfp_test_vars[index])

plot_samples = lfp_psds[index,:,:129]

for i in range(num_plot_samples):
    ax['V'].plot(fs, plot_samples[i], c='black', lw=0.5, alpha=0.8)

upper = mean[:129] + 2 * std[:129]
lower = mean[:129] - 2 * std[:129]

ax['V'].fill_between(fs, lower, upper, color='C1', alpha=0.3)
ax['V'].plot(fs, mean[:129], 'C1', linestyle='--')

ax['V'].set_yticks([0, -8])

ax['V'].set_ylabel('log $P_\phi (\mathbf{r}, f)$', labelpad=0.)
ax['V'].set_xlabel('f (Hz)', labelpad=-1)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Plot marginal standard deviations
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


ax['W'].plot(np.arange(10), hist_stds.mean(0), 'o', color='black', markerfacecolor='none', label='pop. activity')
ax['W'].plot(np.arange(10), lfp_stds.mean(0), 'x', color='grey', label='lfp')
ax['W'].set_xticks(np.arange(10))
ax['W'].set_xticklabels(labels)
ax['W'].set_xticks(np.arange(10))
ax['W'].set_xticklabels(labels)
ax['W'].set_ylim(0., 0.3)
ax['W'].set_ylabel('marginal std.', labelpad=0.5)

errors = np.abs(test_x[:100] - hist_expectations)
lfp_errors = np.abs(test_x[:100] - lfp_expectations)
ax['Z'].plot(np.arange(10), errors.mean(0), 'o', color='black', markerfacecolor='none')
ax['Z'].plot(np.arange(10), lfp_errors.mean(0), 'x', color='grey')
ax['Z'].set_xticks(np.arange(10))
ax['Z'].set_xticklabels(labels)
ax['Z'].set_xticks(np.arange(10))
ax['Z'].set_xticklabels(labels)
ax['Z'].set_ylim(0., 0.3)
ax['Z'].set_ylabel('abs. error', labelpad=0.5)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Plot posterior simulation errors
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

hist_posterior_hist_errors = (hist_psds - test_y[:100,None,:])
lfp_posterior_lfp_errors = (lfp_psds - test_y_lfp[:100,None,:])

hist_posterior_lfp_errors = (hist_posterior_lfp_psds - test_y_lfp[:100,None,:])
lfp_posterior_hist_errors = (lfp_posterior_hist_psds - test_y[:100,None,:])
bins = np.linspace(0., 6., 20)
ax['A'].hist(np.abs(hist_posterior_hist_errors).max(2).flatten(), bins=bins, density=True, color='black', histtype='step', label='simulations pop. activity posterior')
ax['A'].hist(np.abs(lfp_posterior_hist_errors).max(2).flatten(), bins=bins, density=True, color='grey', histtype='step', alpha=0.8, label='simulations LFP posterior')
ax['B'].hist(np.abs(lfp_posterior_lfp_errors).max(2).flatten(), bins=bins, density=True, color='grey', histtype='step')
ax['B'].hist(np.abs(hist_posterior_lfp_errors).max(2).flatten(), bins=bins, density=True, color='black', histtype='step', alpha=0.8)
ax['A'].set_ylim(0, 3.)
ax['B'].set_ylim(0, 3.)
ax['A'].set_xlabel('max abs. error', labelpad=0.)
ax['A'].set_ylabel('density', labelpad=0.)
ax['B'].set_xlabel('max abs. error', labelpad=0.)

labels = ['A', 'B', 'C', 'D']
pos = [(0.08, 0.89), (0.08, 0.59), (0.08, 0.42), (0.08, 0.23)]
for i in range(4):
    tax = fig.add_axes([*pos[i], 0.05, 0.05])
    tax.axis('off')
    tax.text(0, 0, labels[i], fontdict={'fontsize': 12, 'rotation': 0})
ax["A"].legend()
fig.savefig('figure5.pdf', bbox_inches='tight')
plt.show()

