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
Yn = Y - Ytrain_mean

train_x = X[:n_train]
train_y = Yn[:n_train]

test_x = X[n_train:]
test_y = Y[n_train:]


# LFPs
Y_lfp = np.log10(lfp_psds)
Ytrain_lfp_mean = Y_lfp[:n_train].mean(0, keepdims=True)

test_y_lfp = Y_lfp[n_train:]


# Load additional simulations for plotting multiple PSDS from same parameters
sim_path = os.path.join("../simulation_scripts/brunel_simulations_example_prediction_figure")
dirs = sorted(os.listdir(os.path.join(sim_path, "nest_output")))

example_lfp_psds = []
example_hist_psds = []
# loop through directories containing individual simulations
for i, d in enumerate(dirs):
    with open(os.path.join(sim_path, "parameters", d + ".pkl"), "rb") as f:
        param = pickle.load(f)

    with h5py.File(os.path.join(sim_path, "nest_output", d, "LFP_firing_rate.h5"), "r") as f:
        ex_hist = f["ex_hist"][()]
        in_hist = f["in_hist"][()]

    fs, epsd = welch(ex_hist)
    fs, ipsd = welch(in_hist)
    example_hist_psds.append(np.concatenate((epsd, ipsd)))


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
    example_lfp_psds.append(psd.flatten())


# %%


data_dir = os.path.join("..", "train_conditional_distributions", "compute_predictions_errors")
dgp_preds = np.load(os.path.join(data_dir, "dgp_test_mean.npy"))
dgp_var = np.load(os.path.join(data_dir, "dgp_test_var.npy"))
dgp_preds += Ytrain_mean
dgp_errors = np.load(os.path.join(data_dir, "dgp_test_error.npy"))

dgp_preds_lfp = np.load(os.path.join(data_dir, "dgp_lfp_test_mean.npy"))
dgp_var_lfp = np.load(os.path.join(data_dir, "dgp_lfp_test_var.npy"))
dgp_preds_lfp += Ytrain_lfp_mean
dgp_errors_lfp = np.load(os.path.join(data_dir, "dgp_lfp_test_error.npy"))

temp_dir = os.path.join("..", "train_conditional_distributions", "compare_maf_implementation_to_sbi")
maf_preds = np.load(os.path.join(data_dir, "maf_test_mean.npy"))
maf_var = np.load(os.path.join(data_dir, "maf_test_var.npy"))
maf_preds += Ytrain_mean
maf_errors = np.load(os.path.join(data_dir, "maf_test_error.npy"))

#hcpls_preds = np.load(os.path.join("..", "train_pls", "test_preds_pls.npy"))
#hcpls_preds += Ytrain_mean
#hcpls_errors = np.load(os.path.join("..", "train_pls", "test_errors_pls.npy"))

fs = np.linspace(0,500,129)

np.random.seed(123)


# %%

fig = plt.figure()
fig.subplots_adjust(left=0.1, wspace=0.4, right=0.98)
fig.set_size_inches([7,7])
gs = GridSpec(11, 4)

plot_index1 = 4
plot_index2 = 6

#
# Histogram example plot
#

ax11 = fig.add_subplot(gs[:2,:2])

for i in range(10):
    label = "simulation" if i == 0 else None
    ax11.plot(fs, np.log10(example_hist_psds[i][:129]), lw=1., color="black", label=label)

ax11.plot(fs, dgp_preds[plot_index1,:129], lw=2., color="C0", label="DGP")
upper = dgp_preds[plot_index1,:129] + 2 * np.sqrt(dgp_var[plot_index1,:129])
lower = dgp_preds[plot_index1,:129] - 2 * np.sqrt(dgp_var[plot_index1,:129])
ax11.fill_between(fs, lower, upper, color='C0', alpha=0.3)


ax11.plot(fs, maf_preds[plot_index1,:129], lw=2., color="C1", label="MAF")
upper = maf_preds[plot_index1,:129] + 2 * np.sqrt(maf_var[plot_index1,:129])
lower = maf_preds[plot_index1,:129] - 2 * np.sqrt(maf_var[plot_index1,:129])
ax11.fill_between(fs, lower, upper, color='C1', alpha=0.3)

#ax11.plot(fs, hcpls_preds[plot_index1,:129], lw=2., color="C2", label="HCPLS")

ax11.legend()
ax11.set_xticklabels([])


ax12 = fig.add_subplot(gs[2:4,:2])

for i in range(10):
    label = "simulation" if i == 0 else None
    ax12.plot(fs, np.log10(example_hist_psds[10+i][:129]), lw=1., color="black", label=label)

ax12.plot(fs, test_y[plot_index2,:129], lw=3., color="black", label="simulation")

ax12.plot(fs, dgp_preds[plot_index2,:129], lw=2., color="C0", label="DGP")
upper = dgp_preds[plot_index2,:129] + 2 * np.sqrt(dgp_var[plot_index2,:129])
lower = dgp_preds[plot_index2,:129] - 2 * np.sqrt(dgp_var[plot_index2,:129])
ax12.fill_between(fs, lower, upper, color='C0', alpha=0.3)


ax12.plot(fs, maf_preds[plot_index2,:129], lw=2., color="C1", label="MAF")
upper = maf_preds[plot_index2,:129] + 2 * np.sqrt(maf_var[plot_index2,:129])
lower = maf_preds[plot_index2,:129] - 2 * np.sqrt(maf_var[plot_index2,:129])
ax12.fill_between(fs, lower, upper, color='C1', alpha=0.3)

#ax12.plot(fs, hcpls_preds[plot_index2,:129], lw=2., color="C2", label="HCPLS")

ax11.set_ylabel('log $P_{\\nu_E} (f)$', labelpad=-1)
ax12.set_ylabel('log $P_{\\nu_E} (f)$', labelpad=-1)

ax12.set_xlabel("f (Hz)")

ax11.set_title("Pop. spiking activity PSD")
#
# Error histograms
#

bins = np.linspace(0,2,51)
ax2 = fig.add_subplot(gs[5:8,:2])
ax2.hist(np.abs(dgp_errors).max(1), bins=bins, density=True, alpha=0.6, label="DGP")
ax2.hist(np.abs(maf_errors).max(1), bins=bins, density=True, alpha=0.6, label="MAF")
#ax2.hist(np.abs(hcpls_errors).max(1), bins=bins, density=True, alpha=0.6, label="HCPLS")

ax2.set_xlabel("max abs. error")

ax2.set_ylabel("density", labelpad=0)

ax2.set_xlim(0, 1.25)
ax2.legend()


#
# LFP example plot
#

ax31 = fig.add_subplot(gs[:2,2:])
for i in range(10):
    label = "simulation" if i == 0 else None
    ax31.plot(fs, np.log10(example_lfp_psds[i][:129]), lw=1., color="black", label=label)

ax31.plot(fs, dgp_preds_lfp[plot_index1,:129], lw=2., color="C0", label="DGP")

upper = dgp_preds_lfp[plot_index1,:129] + 2 * np.sqrt(dgp_var_lfp[plot_index1,:129])
lower = dgp_preds_lfp[plot_index1,:129] - 2 * np.sqrt(dgp_var_lfp[plot_index1,:129])
ax31.fill_between(fs, lower, upper, color='C0', alpha=0.3)

ax31.legend()
ax31.set_xticklabels([])


ax32 = fig.add_subplot(gs[2:4,2:])

for i in range(10):
    label = "simulation" if i == 0 else None
    ax32.plot(fs, np.log10(example_lfp_psds[10+i][:129]), lw=1., color="black", label=label)


ax32.plot(fs, test_y_lfp[plot_index2,:129], lw=3., color="black", label="simulation")
ax32.plot(fs, dgp_preds_lfp[plot_index2,:129], lw=2., color="C0", label="DGP")

upper = dgp_preds_lfp[plot_index2,:129] + 2 * np.sqrt(dgp_var_lfp[plot_index2,:129])
lower = dgp_preds_lfp[plot_index2,:129] - 2 * np.sqrt(dgp_var_lfp[plot_index2,:129])
ax32.fill_between(fs, lower, upper, color='C0', alpha=0.3)


ax31.set_ylabel('log $P_{\\phi_1} (f)$', labelpad=-1)
ax32.set_ylabel('log $P_{\\phi_1} (f)$', labelpad=-4)

ax31.set_title("LFP PSD")

ax32.set_xlabel("f (Hz)")


#
# Error histograms LFP
#

bins = np.linspace(0,2,51)
ax4 = fig.add_subplot(gs[5:8,2:])
ax4.hist(np.abs(dgp_errors_lfp).max(1), bins=bins, density=True, alpha=0.6, label="DGP")
ax4.set_xlabel("max abs. error")
ax4.legend()

ax4.set_ylabel("density", labelpad=2)
ax4.set_xlim(0, 1.25)
labels = ['A', 'B', 'C', 'D', "E", "F"]
pos = [(0.04, 0.9), (0.52, 0.9), (0.04, 0.53), (0.52, 0.53), (0.04, 0.25), (0.52, 0.25)]
for i in range(6):
    tax = fig.add_axes([*pos[i], 0.05, 0.05])
    tax.axis('off')
    tax.text(0, 0, labels[i], fontdict={'fontsize': 12, 'rotation': 0})

#
# Std MAF/DGP
#

ax5 = fig.add_subplot(gs[9:,:2])
ax5.plot(fs, np.sqrt(dgp_var).mean(0)[:129], color="C0", label="DGP")
ax5.plot(fs, np.sqrt(maf_var).mean(0)[:129], color="C1", label="MAF")

ax5.set_ylim(0.045, 0.22)
ax5.set_xlabel("f (Hz)")
ax5.set_ylabel("mean std")
ax5.legend()

#
# Std DGP LFP
#

ax6 = fig.add_subplot(gs[9:,2:])
ax6.plot(fs, np.sqrt(dgp_var_lfp).mean(0)[:129], color="C0", label="DGP")

ax6.set_ylim(0.045, 0.22)
ax6.legend()
ax6.set_xlabel("f (Hz)")
ax6.set_ylabel("mean std", labelpad=2)

fig.savefig("figure3.pdf")
plt.show()





