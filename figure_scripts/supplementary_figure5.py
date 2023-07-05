import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.gridspec import GridSpec
import numpy as np
import os, pickle, h5py

plt.rcParams["font.size"] = 7

topdir = os.path.join("..", "simulation_scripts", "brunel_simulations_lfp_validation")

ex_kernels_interp = []
in_kernels_interp = []
for i in range(5):
    ek = []
    ik = []
    for j in range(6):
        with h5py.File(os.path.join("..", "simulation_scripts", "kernel_sims", "delta_kernels",
            f"delta_kernel_{i}_{j}.h5"), "r") as f:
            ek.append(f["ex"][()])
            ik.append(f["in"][()])
    ex_kernels_interp.append(ek)
    in_kernels_interp.append(ik)

ex_kernels_interp = np.array(ex_kernels_interp).transpose((1,0,2,3)) # (tauMem, CMem, channel, time)
in_kernels_interp = np.array(in_kernels_interp).transpose((1,0,2,3))

def compute_delta_kernel(tau_m, c_m):
    tauMems = np.array([15., 18., 21., 24., 27., 30.])
    CMems = np.array([0.4, 0.6, 0.8, 1.0, 1.2]) # values used in hybridLFPy

    ex_kernel = np.zeros((6,1001), dtype=np.float64)
    in_kernel = np.zeros((6,1001), dtype=np.float64)
    for j in range(6):
        for i in range(1001):
            interp = RegularGridInterpolator((tauMems, CMems),  ex_kernels_interp[:,:,j,i])
            ex_kernel[j,i] = interp((tau_m, c_m))

            interp = RegularGridInterpolator((tauMems, CMems),  in_kernels_interp[:,:,j,i])
            in_kernel[j,i] = interp((tau_m, c_m))
    return ex_kernel, in_kernel

# Alpha function for computing LFPs
def isyn_fn(J, tau):
    t = np.arange(0, 100, 0.1)
    return - J * t/tau * np.exp(1 - t/tau) * 1e-3


def plot_lfps(index, ax):
    with open(os.path.join(topdir, "parameters", f"{index:05d}.pkl"), "rb") as f:
        param = pickle.load(f)

    with h5py.File(os.path.join(topdir, "nest_output", f"{index:05d}", "LFP_firing_rate.h5")) as f:
        ex_hist = f["ex_hist"][()]
        in_hist = f["in_hist"][()]

    with h5py.File(os.path.join(topdir, "hybrid_output", f"{index:05d}", "lfp_sum.h5")) as f:
        lfp = f["data"][()]
    lfp = lfp[:,:20000].reshape((6,-1,10)).mean(-1)


    ex_isyn = isyn_fn(param["J"], param["tauSyn"])
    in_isyn = isyn_fn(-param["g"] * param["J"], param["tauSyn"])
    ex_kernel_delta, in_kernel_delta = compute_delta_kernel(param["tauMem"], param["CMem"] / 250.)
    ex_kernel = []
    in_kernel = []
    for j in range(6):
        ex_kernel.append(np.convolve(ex_isyn, ex_kernel_delta[j]))
        in_kernel.append(np.convolve(in_isyn, in_kernel_delta[j]))

    ex_kernel = np.array(ex_kernel).reshape((6,200,10)).mean(axis=-1)
    in_kernel = np.array(in_kernel).reshape((6,200,10)).mean(axis=-1)

    lfp_approx = []
    for j in range(6):
        lfp_approx.append(np.convolve(ex_hist, ex_kernel[j], mode="full") + np.convolve(in_hist, in_kernel[j], mode="full"))
    lfp_approx = np.array(lfp_approx)

    lfp = lfp[:,200:1500]
    lfp_approx = lfp_approx[:,200:1700]

    lfp -= lfp.mean(1, keepdims=True)
    lfp_approx -= lfp_approx.mean(1, keepdims=True)

    # account for delay
    lfp_approx = lfp_approx[:,6:]

    # account for arbitrary scale of LFP
    prop_consts = lfp.std(1) / lfp_approx.std(1)
    prop_signs = np.array([1, 1, 1, -1, 1, 1])
    lfp_approx = lfp_approx * prop_consts[:,None] * prop_signs[:,None]

    # make sure we have no weird edge effects
    lfp = lfp[:,:500]
    lfp_approx = lfp_approx[:,:500]

    maxdiff = (lfp.max(1) - lfp.min(1)).max()
    t = np.arange(lfp.shape[1])
    for i in range(6):
        ch = lfp[i]
        ch = (ch - ch.min())
        approx_ch = lfp_approx[i]
        approx_ch = (approx_ch - approx_ch.min())

        ax.plot(t, ch + 0.9 * (5-i) * maxdiff, label="simulation" if i == 0 else None, color="black")
        ax.plot(t, approx_ch + 0.9 * (5-i) * maxdiff, label="approximation" if i == 0 else None, color="C1", linestyle="--")
    ax.set_xlabel("t (ms)")
    if index == 0:
        ylabels = [f"ch{i}" for i in range(1,7)]
        ypos = ax.get_lines()[::2]
        ypos = [yp.get_ydata()[0] for yp in ypos]
        ax.set_yticks(ypos)
        ax.set_yticklabels(ylabels)
    else:
        ax.set_yticks([])



# %%

fig, ax = plt.subplots(ncols=3, sharex=True, sharey=False)
fig.set_size_inches([7.5,4])
fig.subplots_adjust(left=0.05, right=0.98)
plot_lfps(0, ax[0])
plot_lfps(1, ax[1])
plot_lfps(2, ax[2])
ax[0].legend()
fig.savefig("lfp_validation_figure.pdf")
plt.show()

