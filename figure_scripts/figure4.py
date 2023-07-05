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

n_train_vals = [100, 200, 400, 1000, 2000, 4000, 8000]

data_dir = os.path.join("..", "train_conditional_distributions", "compute_predictions_errors")
test_preds = []
test_errors = []
train_errors = []
for i, n_train in enumerate(n_train_vals[:-1]):
    Ytrain_mean = Y[:n_train].mean(0, keepdims=True)

    #     train_x = X[:n_train]
    #     train_y = Yn[:n_train]
    # 
    #     test_x = X[n_train:]
    #     test_y = Y[n_train:]


    test_preds.append(np.load(os.path.join(data_dir, f"dgp_test_mean_{n_train}.npy")))
    test_preds[-1] += Ytrain_mean
    test_errors.append(np.load(os.path.join(data_dir, f"dgp_test_error_{n_train}.npy")))
    train_errors.append(np.load(os.path.join(data_dir, f"dgp_train_error_{n_train}.npy")))

test_preds.append(np.load(os.path.join(data_dir, f"dgp_test_mean.npy")))
test_errors.append(np.load(os.path.join(data_dir, f"dgp_test_error.npy")))
train_errors.append(np.load(os.path.join(data_dir, f"dgp_train_error.npy")))

mean_abs_test_errors = [np.abs(v).max(1).mean() for v in test_errors]
mean_abs_train_errors = [np.abs(v).max(1).mean() for v in train_errors]

# %%

gs = GridSpec(1,1)
fig = plt.figure()
fig.set_size_inches([7,3])
ax = fig.add_subplot(gs[0,0])
fig.subplots_adjust(left=0.1, bottom=0.15)

ax.plot(n_train_vals, mean_abs_train_errors, "-*", label="test data")
ax.plot(n_train_vals, mean_abs_test_errors, "-*", label="training data")

ax.set_ylabel("Mean max error")
ax.set_xlabel("# training samples")

ax.legend()

fig.savefig("figure4.pdf")
plt.show()

