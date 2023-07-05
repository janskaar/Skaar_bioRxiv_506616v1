import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from functools import partial
import os, pickle, h5py


n_train = 8000

sim_path = os.path.join("../simulation_scripts/brunel_simulations_10p")
dirs = sorted(os.listdir(os.path.join(sim_path, "nest_output")))

hists = []
params = []
# loop through directories containing individual simulations
for i, d in enumerate(dirs):
    with open(os.path.join(sim_path, "parameters", d + ".pkl"), "rb") as f:
        param = pickle.load(f)
        params.append(np.array([param["eta"], param["g"], param["tauSyn"], param["tauMem"], param["delay"], param["t_ref"], param["cSyn"], param["theta"], param["CMem"], param["V_reset"]]))

    with h5py.File(os.path.join(sim_path, "nest_output", d, "LFP_firing_rate.h5"), "r") as f:
        ex_hist = f["ex_hist"][()]
        in_hist = f["in_hist"][()]
    hists.append(np.stack((ex_hist, in_hist)))

hists = np.array(hists)
params = np.array(params)

# remove simulations in which activity is strongly synchronous
args = ((hists[:,0,:] > 800).sum(axis=1) > 150) & ((hists[:,0,:] < 20).sum(axis=1) > 500)

labels = params[~args].copy()

plt.rcParams['font.size'] = 6

minlabels = labels.min(axis=0)
difflabels = labels.max(axis=0) - labels.min(axis=0)
labels -= minlabels
labels /= difflabels

X = labels.copy().astype(np.float32)

test_x = X[n_train:]
index = 6
x0 = test_x[index]

dgp_sample = np.load(os.path.join("..", "train_posterior_distributions", "mcmc_posterior_samples_dgp", f"{index:04d}", 'proposals.npy'))
num_chains = dgp_sample.shape[1]
acceptance = np.load(os.path.join("..", "train_posterior_distributions", "mcmc_posterior_samples_dgp", f"{index:04d}", 'acceptance.npy'))
dgp_sample = np.concatenate([dgp_sample[:,i,:][acceptance[:,i]] for i in range(num_chains)])[::10]

lfp_sample = np.load(os.path.join("..", "train_posterior_distributions", "mcmc_posterior_samples_lfp_2ch_dgp", f"{index:04d}", 'proposals.npy'))
num_chains = lfp_sample.shape[1]
acceptance = np.load(os.path.join("..", "train_posterior_distributions", "mcmc_posterior_samples_lfp_2ch_dgp", f"{index:04d}", 'acceptance.npy'))
lfp_sample = np.concatenate([lfp_sample[:,i,:][acceptance[:,i]] for i in range(num_chains)])[::10]


# %%


xticklabels = [('1.0', '3.5'), ('4.5', '8.0'), ('1.0', '8.0'), ('15', '30'), ('0.1', '3.9'), ('0.0', '5.0'), ('25', '100'), ('15', '25'), ('100', '300'), ('0', '10')]
xx, yy = np.meshgrid(*[np.linspace(0,1,50, dtype=np.float32)]*2)
xgrid = np.dstack([xx, yy]).reshape((-1,2))
fig, ax = plt.subplots(ncols=21, nrows=10, sharex=False, sharey=False)
fig.set_size_inches([7.5, 4.])

for i in range(10):
    ax[i,10].axis("off")

#
# LFP POSTERIOR
#
lfp_ax = ax[:,11:]

for i in range(1,10):
    for j in range(i):
        lfp_ax[i,j].axis('off')

stds = lfp_sample.std(axis=0)
for i in range(10):
    for j in range(i+1,10):
        bw = np.mean([stds[i], stds[j]]) * 0.33
        kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(lfp_sample[:,[i, j]])
        gridprob = np.exp(kde.score_samples(xgrid))
        lfp_ax[i,j].pcolormesh(xx, yy, gridprob.reshape((50,50)).T, vmax=gridprob.max(), vmin=0., shading='auto')
        lfp_ax[i,j].scatter(x0[j], x0[i], c='red', s=10.)
        lfp_ax[i,j].set_xlim(0,1)
        lfp_ax[i,j].set_ylim(0,1)
        lfp_ax[i,j].set_xticks([])
        lfp_ax[i,j].set_yticks([])
xlabels = ['$\\eta$', '$g$', '$\\tau_s$', '$\\tau_m$', 'delay', '$t_{ref}$', '$Q_s$', '$\\theta$', '$C_{mem}$', '$V_{reset}$']
x = np.linspace(0,1,100)[:,None]
for i in range(10):
    bw = stds[i] * 0.33
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(lfp_sample[:,i:i+1])
    prob = np.exp(kde.score_samples(x))
    lfp_ax[i,i].plot(x, prob, 'black')
    lfp_ax[i,i].vlines(x0[i], 0, prob.max(), color='red')
    lfp_ax[i,i].set_xlim(0,1)
    lfp_ax[i,i].set_xlabel(xlabels[i], labelpad=-1.5)
    lfp_ax[i,i].set_yticks([])
    lfp_ax[i,i].set_xticks([0, 1])
    lfp_ax[i,i].set_xticklabels(xticklabels[i])

#
# Pop. act. POSTERIOR
#
hist_ax = ax[:,:10]

for i in range(1,10):
    for j in range(i):
        hist_ax[i,j].axis('off')

stds = dgp_sample.std(axis=0)
for i in range(10):
    for j in range(i+1,10):
        bw = np.mean([stds[i], stds[j]]) * 0.33
        kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(dgp_sample[:,[i, j]])
        gridprob = np.exp(kde.score_samples(xgrid))
        hist_ax[i,j].pcolormesh(xx, yy, gridprob.reshape((50,50)).T, vmax=gridprob.max(), vmin=0., shading='auto')
        hist_ax[i,j].scatter(x0[j], x0[i], c='red', s=10.)
        hist_ax[i,j].set_xlim(0,1)
        hist_ax[i,j].set_ylim(0,1)
        hist_ax[i,j].set_xticks([])
        hist_ax[i,j].set_yticks([])
xlabels = ['$\\eta$', '$g$', '$\\tau_s$', '$\\tau_m$', 'delay', '$t_{ref}$', '$Q_s$', '$\\theta$', '$C_{mem}$', '$V_{reset}$']
x = np.linspace(0,1,100)[:,None]
for i in range(10):
    bw = stds[i] * 0.33
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(dgp_sample[:,i:i+1])
    prob = np.exp(kde.score_samples(x))
    hist_ax[i,i].plot(x, prob, 'black')
    hist_ax[i,i].vlines(x0[i], 0, prob.max(), color='red')
    hist_ax[i,i].set_xlim(0,1)
    hist_ax[i,i].set_xlabel(xlabels[i], labelpad=-1.5)
    hist_ax[i,i].set_yticks([])
    hist_ax[i,i].set_xticks([0, 1])
    hist_ax[i,i].set_xticklabels(xticklabels[i])


titleax1 = fig.add_axes([0.7, 0.9, 0.1, 0.05])
titleax2 = fig.add_axes([0.2, 0.9, 0.1, 0.05])
titleax1.axis("off")
titleax2.axis("off")
titleax1.text(0, 0, "LFP", fontsize=12)
titleax2.text(0, 0, "Pop. spiking activity", fontsize=12)
fig.savefig("supplementary_fig3.pdf")
plt.show()
