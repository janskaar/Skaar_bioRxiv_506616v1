from deep_lmc_natural_gradients import DeepLMC
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
import gpytorch, torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from scipy.signal import welch
import os, time, pickle, h5py
import train_likelihood
from functools import partial
import jax
from jax.scipy.interpolate import RegularGridInterpolator

BATCH_SIZE = 100
EVAL_BATCH_SIZE = 5

device = "cuda"

n_train = 8000


sim_path = os.path.join("../../simulation_scripts/brunel_simulations_10p")
dirs = sorted(os.listdir(os.path.join(sim_path, "nest_output")))


# Load kernels and define functions to compute LFPs
ex_kernels_interp = []
in_kernels_interp = []
for i in range(5):
    ek = []
    ik = []
    for j in range(6):
        with h5py.File(f"delta_kernels/delta_kernel_{i}_{j}.h5", "r") as f:
            ek.append(f["ex"][()])
            ik.append(f["in"][()])
    ex_kernels_interp.append(ek)
    in_kernels_interp.append(ik)

ex_kernels_interp = np.array(ex_kernels_interp).transpose((1,0,2,3)) # (tauMem, CMem, channel, time)
in_kernels_interp = np.array(in_kernels_interp).transpose((1,0,2,3))

def compute_delta_kernel(tau_m, c_m, kernel_ct):
    # kernel_ct = kernel at channel c and time t
    tauMems = np.array([15., 18., 21., 24., 27., 30.])
    CMems = np.array([0.4, 0.6, 0.8, 1.0, 1.2]) # values used in hybridLFPy
    interp = RegularGridInterpolator((tauMems, CMems),  kernel_ct)
    return interp((tau_m, c_m))

# vmap over time and channels
compute_delta_kernel_vmap = jax.vmap(compute_delta_kernel, in_axes=(None,None,3))
compute_delta_kernel_vmap = jax.vmap(compute_delta_kernel, in_axes=(None,None,2))

def isyn_fn(J, tau):
    t = np.arange(0, 100, 0.1)
    return - J * t/tau * np.exp(1 - t/tau) * 1e-3

# loop through directories containing individual simulations
hists = []
lfps = []
params = []
for i, d in enumerate(dirs):
    print(i, end="\r")
    with open(os.path.join(sim_path, "parameters", d + ".pkl"), "rb") as f:
        param = pickle.load(f)
    params.append(np.array([param["eta"], param["g"], param["tauSyn"], param["tauMem"], param["delay"], param["t_ref"], param["cSyn"], param["theta"], param["CMem"], param["V_reset"]]))

    ex_kernel_delta = compute_delta_kernel_vmap(param["tauMem"], param["CMem"] / 250., ex_kernels_interp)
    in_kernel_delta = compute_delta_kernel_vmap(param["tauMem"], param["CMem"] / 250., in_kernels_interp)
    ex_kernel = []
    in_kernel = []
    ex_isyn = isyn_fn(param['J'], param['tauSyn'])
    in_isyn = isyn_fn(-param['J'] * param['g'], param['tauSyn'])
    for j in range(6):
        ex_kernel.append(np.convolve(ex_isyn, ex_kernel_delta[j]))
        in_kernel.append(np.convolve(in_isyn, in_kernel_delta[j]))
    ex_kernel = np.array(ex_kernel).reshape((6,200,10)).mean(axis=-1)
    in_kernel = np.array(in_kernel).reshape((6,200,10)).mean(axis=-1)

    with h5py.File(os.path.join(sim_path, "nest_output", d, "LFP_firing_rate.h5"), "r") as f:
        ex_hist = f["ex_hist"][()]
        in_hist = f["in_hist"][()]
    lfp = []
    for j in range(6):
        lfp.append(np.convolve(ex_hist[501:], ex_kernel[j], mode='full') + np.convolve(in_hist[501:], in_kernel[j], mode='full'))
    lfp = np.array(lfp)[...,500:] # remove first 500 to get rid of boundary effects of convolution
    lfp -= lfp.mean(1, keepdims=True)
    lfps.append(lfp)
    hists.append(np.stack((ex_hist, in_hist)))


hists = np.array(hists)
lfps = np.array(lfps)
params = np.array(params)

np.save("hists.npy", hists)
np.save("lfps.npy", hists)

# remove simulations in which activity is strongly synchronous
args = ((hists[:,0,:] > 800).sum(axis=1) > 150) & ((hists[:,0,:] < 20).sum(axis=1) > 500)

labels = params[~args].copy()
fs, psds = welch(lfps[~args,:,501:])

psds = psds.reshape((-1, 129*6))

minlabels = np.array([1.0, 4.5, 1., 15., 0.1, 0.0, 25., 15., 100., 0.])
difflabels = np.array([2.5, 3.5, 7., 15., 3.9, 5., 75., 10., 200., 10.])
labels -= minlabels
labels /= difflabels

X = torch.tensor(labels.copy().astype(np.float32))
Y = torch.tensor(np.log10(psds.copy().astype(np.float32)))
Ytrain_mean = Y[:n_train].mean(0, keepdims=True)
Yn = Y - Ytrain_mean

train_x = torch.Tensor(X[:n_train]).to(device=device)
train_y = torch.Tensor(Yn[:n_train]).to(device=device)

test_x = torch.Tensor(X[n_train:]).to(device=device)
test_y = torch.Tensor(Yn[n_train:]).to(device=device)

print(f"train_x shape: {train_x.shape}", flush=True)
print(f"test_x shape: {test_x.shape}", flush=True)
print(f"train_y shape: {train_y.shape}", flush=True)
print(f"test_y shape: {test_y.shape}", flush=True)


train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TensorDataset(test_x, test_y)
eval_test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
eval_train_loader = DataLoader(train_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

num_tasks = train_y.size(-1)
num_inducing1 = 128
num_inducing2 = 128

def train_batch_dgp(x_batch, y_batch, model, mll, hyp_opt, var_opt):
     variational_ngd_optimizer.zero_grad()
     hyperparameter_optimizer.zero_grad()
     output = model(x_batch)
     loss = -mll(output, y_batch)
     loss.backward()
     variational_ngd_optimizer.step()
     hyperparameter_optimizer.step()

def eval_batch_dgp(x_batch, y_batch, model):
    # reshape to (num, 1, output_features), so we
    # compute distribution independently for each example
    dist = model.likelihood(model.memory_efficient_call(x_batch[:,None,:]))
    log_probs = dist.log_prob(y_batch[None,:,None,:])
    log_probs = torch.logsumexp(log_probs, 0) - torch.log(torch.tensor(log_probs.shape[0]).to(device))
    return -log_probs.mean().cpu().item()

lr = 0.0001
torch.random.manual_seed(1234)
model = DeepLMC(10,
        num_tasks=6*129,
        num_hidden_dgp_dims=20,
        num_latent_gps=50,
        num_inducing1=num_inducing1,
        num_inducing2=num_inducing2).to(device)
 
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))
mll = mll.to(device)
variational_lr = 1e-3
variational_ngd_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=train_y.size(0), lr=variational_lr)
hyperparameter_optimizer = torch.optim.Adam([
   {"params": model.hyperparameters()}], lr=lr)

train_batch_dgp_partial = partial(train_batch_dgp,
                                  model=model,
                                  mll=mll, 
                                  hyp_opt=hyperparameter_optimizer,
                                  var_opt=variational_ngd_optimizer)
eval_batch_dgp_partial = partial(eval_batch_dgp, model=model)

train_likelihood.STOP_PERSISTENCE = 10
train_likelihood.EVALUATION_FREQUENCY = 2
train_likelihood.train(
    model,
    train_batch_dgp_partial,
    eval_batch_dgp_partial,
    train_loader,
    eval_train_loader,
    eval_test_loader,
    logfile=f"dgp_lfp_20_50_00001.log",
    savefile=f"dgp_lfp_20_50_00001.model"
    )

