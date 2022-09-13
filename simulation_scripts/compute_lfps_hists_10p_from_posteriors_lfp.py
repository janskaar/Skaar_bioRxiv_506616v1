import os, h5py, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

sim_path = os.path.join('./brunel_simulations_10p')

# Kernel for computing LFP from population spiking activities
with h5py.File('lfp_kernel_delta.h5', 'r') as f:
    ex_kernel_delta = f['ex'][()]
    in_kernel_delta = f['in'][()]

# Alpha function for computing LFPs
def isyn_fn(J, tau):
    t = np.arange(0, 100, 0.1)
    return - J * t/tau * np.exp(1 - t/tau) * 1e-3

dirs = sorted(os.listdir(os.path.join(sim_path, 'nest_output')))
lfps = []
params = []
ehists = []
ihists = []

# Loop through directories containing individual simulations
for i, d in enumerate(dirs):
    if not os.path.isfile(os.path.join(sim_path, 'nest_output', d, 'LFP_firing_rate.h5')):
        continue
    with open(os.path.join(sim_path, 'parameters', d + '.pkl'), 'rb') as f:
        param = pickle.load(f)

    ex_kernel = []
    in_kernel = []
    ex_isyn = isyn_fn(param['J'], param['tauSyn'])
    in_isyn = isyn_fn(-param['J'] * param['g'], param['tauSyn'])
    for i in range(6):
        ex_kernel.append(np.convolve(ex_isyn, ex_kernel_delta[i]))
        in_kernel.append(np.convolve(in_isyn, in_kernel_delta[i]))
    ex_kernel = np.array(ex_kernel).reshape((6,200,10)).mean(axis=-1)
    in_kernel = np.array(in_kernel).reshape((6,200,10)).mean(axis=-1)

    with h5py.File(os.path.join(sim_path, 'nest_output', d, 'LFP_firing_rate.h5'), 'r') as f:
        ex_hist = f['ex_hist'][()]
        in_hist = f['in_hist'][()]
        lfp = []
        for i in range(6):
            lfp.append(np.convolve(ex_hist, ex_kernel[i], mode='valid') + np.convolve(in_kernel[i], in_hist, mode='valid'))
    lfps.append(lfp)
    params.append(param)
    ehists.append(ex_hist)
    ihists.append(in_hist)

lfps = np.array(lfps)
ps = []
for param in params:
    p = [param['eta'], param['g'], param['tauSyn'], param['tauMem'], param['delay'], param['t_ref'], param['cSyn'], param['theta'], param['CMem'], param['V_reset']]
    ps.append(p)
ps = np.array(ps)
hists = np.array([ehists, ihists]).transpose((1,0,2))

# Compute PSDS, discard first 501 time due to start-up effects
fs, lfppsds = welch(lfps[:,:,501:])
fs, histpsds = welch(hists[:,:,501:])

np.save('posterior_simulations_lfp.npy', histpsds)
np.save('posterior_simulations_lfp_rate.npy', lfppsds)
np.save('posterior_simulations_lfp_labels.npy', ps)
