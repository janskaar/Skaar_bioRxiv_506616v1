import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.signal import welch
import os, pickle, h5py

plt.rcParams["font.size"] = 7

n_train = 8000

sim_path = os.path.join("../simulation_scripts/brunel_simulations_10p")
dirs = sorted(os.listdir(os.path.join(sim_path, "nest_output")))

with h5py.File('lfp_kernel_delta.h5', 'r') as f:
    ex_kernel_delta = f['ex'][()]
    in_kernel_delta = f['in'][()]

# Alpha function for computing LFPs
def isyn_fn(J, tau):
    t = np.arange(0, 100, 0.1)
    return - J * t/tau * np.exp(1 - t/tau) * 1e-3

n_train = 8000

# Load main data set
sim_path = os.path.join("../simulation_scripts/brunel_simulations_10p")
dirs = sorted(os.listdir(os.path.join(sim_path, "nest_output")))

params = []
hists = []
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


# %%


data_dir = os.path.join("..", "train_conditional_distributions", "compute_predictions_errors")
dgp_preds = np.load(os.path.join(data_dir, "dgp_test_mean.npy"))
dgp_var = np.load(os.path.join(data_dir, "dgp_test_var.npy"))
dgp_preds += Ytrain_mean
dgp_errors = np.load(os.path.join(data_dir, "dgp_test_error.npy"))

temp_dir = os.path.join("..", "train_conditional_distributions", "compare_maf_implementation_to_sbi")
maf_preds = np.load(os.path.join(data_dir, "maf_test_mean.npy"))
maf_var = np.load(os.path.join(data_dir, "maf_test_var.npy"))
maf_preds += Ytrain_mean
maf_errors = np.load(os.path.join(data_dir, "maf_test_error.npy"))

hcpls_preds = np.load(os.path.join("..", "train_pls", "test_preds_pls.npy"))
hcpls_preds += Ytrain_mean
hcpls_errors = np.load(os.path.join("..", "train_pls", "test_errors_pls.npy"))

fs = np.linspace(0,500,129)

np.random.seed(123)

# %%

global_comps = [2, 4, 6, 8, 10]
local_comps = [10, 15, 20, 25, 30, 35, 40, 45, 50]
clusters = [2, 4, 6, 8, 10, 12]

hyperparam_errs = np.load(os.path.join("..", "train_pls", "rmaxe_arr.npy"))

fig = plt.figure()
fig.subplots_adjust(left=0.1, bottom=0.2, right=0.98)
fig.set_size_inches([7,4])
gs = GridSpec(1,2)

#
# Error histograms
#

bins = np.linspace(0,2,51)
ax1 = fig.add_subplot(gs[0,0])
ax1.hist(np.abs(dgp_errors).max(1), bins=bins, density=True, alpha=0.6, label="DGP")
ax1.hist(np.abs(maf_errors).max(1), bins=bins, density=True, alpha=0.6, label="MAF")
ax1.hist(np.abs(hcpls_errors).max(1), bins=bins, density=True, alpha=0.6, label="HCPLS")

ax1.set_xlabel("max abs. error")

ax1.set_ylabel("density", labelpad=0)

ax1.set_xlim(0, 1.25)
ax1.legend()
ax1.set_title("test errors")

ax2 = fig.add_subplot(gs[0,1])
ax2.pcolormesh(hyperparam_errs[0])
ax2.set_yticks(np.arange(len(local_comps)) + 0.5)
ax2.set_yticklabels(local_comps)
ax2.set_xticks(np.arange(len(clusters)) + 0.5)
ax2.set_xticklabels(clusters)
ax2.set_xlabel("num clusters")
ax2.set_ylabel("num local comps")
ax2.set_title("test errors")

color = "white"
for y, _ in enumerate(local_comps):
    for x, _ in enumerate(clusters):
        val = hyperparam_errs[0,y,x]
        ax2.text(x + 0.5, y + 0.5, f"{val:.3f}",
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=6,
                   color=color)


fig.savefig("pls_figure.pdf")
plt.show()





