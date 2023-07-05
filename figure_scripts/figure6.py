import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.signal import welch
import os, pickle, h5py

plt.rcParams["font.size"] = 7

n_train = 8000

# Load main data set
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
    hists.append(np.stack((ex_hist, in_hist)))

print("FINISHED LOADING")

hist_psds = np.array(hist_psds)
hists = np.array(hists)
params = np.array(params)

# remove simulations in which activity is strongly synchronous
args = ((hists[:,0,:] > 800).sum(axis=1) > 150) & ((hists[:,0,:] < 20).sum(axis=1) > 500)

labels = params[~args].copy()
hist_psds = hist_psds[~args]

minlabels = np.array([1.0, 4.5, 1., 15., 0.1, 0.0, 25., 15., 100., 0.])
difflabels = np.array([2.5, 3.5, 7., 15., 3.9, 5., 75., 10., 200., 10.])
labels -= minlabels
labels /= difflabels

X = labels.copy().astype(np.float32)
Y = np.log10(hist_psds.copy().astype(np.float32))
Ytrain_mean = Y[:n_train].mean(0, keepdims=True)
Yn = Y - Ytrain_mean

train_x = X[:n_train]
train_y = Yn[:n_train]

test_x = X[n_train:]
test_y = Y[n_train:]

# DGP posterior distributions
dgp_posterior_dir = os.path.join('../train_posterior_distributions/mcmc_posterior_samples_dgp')

dgp_stds = []
dgp_expectations = []
for i in range(100):
    d = os.path.join(dgp_posterior_dir, f'{i:04d}')
    sample = np.load(os.path.join(d, 'proposals.npy'))
    num_chains = sample.shape[1]
    acceptance = np.load(os.path.join(d, 'acceptance.npy'))
    sample  = np.concatenate([sample[:,i,:][acceptance[:,i]] for i in range(num_chains)])
    dgp_stds.append(sample.std(0))
    dgp_expectations.append(sample.mean(0))
#    if i == index:
#        dgp_sample = sample
dgp_stds = np.array(dgp_stds)
dgp_expectations = np.array(dgp_expectations)
dgp_errors_post = np.abs(dgp_expectations - test_x[:100])


# MAF posterior distributions
maf_posterior_dir = os.path.join('../train_posterior_distributions/mcmc_posterior_samples_maf')

maf_stds = []
maf_expectations = []
for i in range(100):
    d = os.path.join(maf_posterior_dir, f'{i:04d}')
    sample = np.load(os.path.join(d, 'proposals.npy'))
    num_chains = sample.shape[1]
    acceptance = np.load(os.path.join(d, 'acceptance.npy'))
    sample  = np.concatenate([sample[:,i,:][acceptance[:,i]] for i in range(num_chains)])
    maf_stds.append(sample.std(0))
    maf_expectations.append(sample.mean(0))
#    if i == index:
#        hist_sample = sample
maf_stds = np.array(maf_stds)
maf_expectations = np.array(maf_expectations)
maf_errors_post = np.abs(maf_expectations - test_x[:100])

# load simulations from DGP posterior
sim_path = os.path.join("../simulation_scripts/brunel_simulations_dgp_posterior_mcmc")
dirs = sorted(os.listdir(os.path.join(sim_path, "nest_output")))
dgp_psds = []
# loop through directories containing individual simulations
for i, d in enumerate(dirs):
    with h5py.File(os.path.join(sim_path, "nest_output", d, "LFP_firing_rate.h5"), "r") as f:
        ex_hist = f["ex_hist"][()]
        in_hist = f["in_hist"][()]

    fs, epsd = welch(ex_hist)
    fs, ipsd = welch(in_hist)
    dgp_psds.append(np.concatenate((epsd, ipsd)))

dgp_psds = np.log10(np.array(dgp_psds)).reshape((100,50,-1))
dgp_errors = np.abs(dgp_psds - test_y[:100][:,None,:])


# load simulations from MAF posterior
sim_path = os.path.join("../simulation_scripts/brunel_simulations_maf_posterior_mcmc")
dirs = sorted(os.listdir(os.path.join(sim_path, "nest_output")))
maf_psds = []
# loop through directories containing individual simulations
for i, d in enumerate(dirs):
    with h5py.File(os.path.join(sim_path, "nest_output", d, "LFP_firing_rate.h5"), "r") as f:
        ex_hist = f["ex_hist"][()]
        in_hist = f["in_hist"][()]

    fs, epsd = welch(ex_hist)
    fs, ipsd = welch(in_hist)
    maf_psds.append(np.concatenate((epsd, ipsd)))

maf_psds = np.log10(np.array(maf_psds)).reshape((100,50,-1))
maf_errors = np.abs(maf_psds - test_y[:100][:,None,:])

# %%

gs = GridSpec(2,2)

labels = ['$\\eta$', '$g$', '$\\tau_s$', '$\\tau_m$', 'delay', '$t_{ref}$', '$Q_s$', '$\\theta$', '$C_{mem}$', '$V_{reset}$']
fig = plt.figure()
fig.set_size_inches([7, 3.5])
fig.subplots_adjust(hspace=0.4, left=0.08, right=0.99)
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(np.arange(10), dgp_stds.mean(0), 'o', color='C0', markerfacecolor='none', label='DGP')
ax1.plot(np.arange(10), maf_stds.mean(0), 'x', color='C1', markerfacecolor='none', label='MAF')
ax1.legend()
ax1.set_xticks(np.arange(10))
ax1.set_xticklabels(labels)
ax1.set_ylabel("marginal std.")
ax1.set_title("Posterior distribution")
ax1.set_ylim(0, 0.28)


ax2 = fig.add_subplot(gs[0,1])
ax2.plot(np.linspace(0,500,129), dgp_psds.std(1).mean(0)[:129], lw=2., color='C0', markerfacecolor='none', label='DGP')
ax2.plot(np.linspace(0,500,129), maf_psds.std(1).mean(0)[:129], lw=2., color='C1', markerfacecolor='none', label='MAF')
ax2.legend()
ax2.set_xlabel("f (Hz)")
ax2.set_ylabel("marginal std.")
ax2.set_title("Posterior predictive distribution")
ax2.set_ylim(0.05,0.23)

ax3 = fig.add_subplot(gs[1,0])
ax3.plot(np.arange(10), dgp_errors_post.mean(0), 'o', color='C0', markerfacecolor='none', label='DGP')
ax3.plot(np.arange(10), maf_errors_post.mean(0), 'x', color='C1', markerfacecolor='none', label='MAF')
ax3.legend()
ax3.set_xticks(np.arange(10))
ax3.set_xticklabels(labels)
ax3.set_ylabel("abs. errors")
ax3.set_ylim(0, 0.28)

bins = np.linspace(0,1,21)
ax4 = fig.add_subplot(gs[1,1])
ax4.plot(np.linspace(0,500,129), dgp_errors.mean((0,1))[:129], color="C0", lw=2., label="DGP")
ax4.plot(np.linspace(0,500,129), maf_errors.mean((0,1))[:129], color="C1", lw=2., label="MAF")
ax4.set_ylabel("abs. errors")
ax4.legend()
ax4.set_xlabel("f (Hz)")
ax4.set_ylim(0.05,0.23)

labels = ['A', 'B', 'C', 'D']
pos = [(0.04, 0.92), (0.52, 0.92), (0.04, 0.44), (0.52, 0.44)]
for i in range(4):
    tax = fig.add_axes([*pos[i], 0.05, 0.05])
    tax.axis('off')
    tax.text(0, 0, labels[i], fontdict={'fontsize': 12, 'rotation': 0})


fig.savefig("figure6.pdf")
plt.show()

