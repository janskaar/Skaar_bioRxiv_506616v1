from deep_lmc_natural_gradients import DeepLMC
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
import gpytorch, torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from scipy.signal import welch
import os, time, pickle, h5py
import train_likelihood
from functools import partial

BATCH_SIZE = 100
EVAL_BATCH_SIZE = 10

device = "cuda"

n_train = 8000

sim_path = os.path.join("../../simulation_scripts/brunel_simulations_10p")
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
fs, psds = welch(hists[~args,:,501:])
psds = psds.reshape((-1, 258))

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

hidden_dims = [40]
num_latent_gps = [10, 20, 30, 40, 50, 60]
seeds = [16]
lr = 0.01
for i, hidden_dim in enumerate(hidden_dims):
    for j, num_latent in enumerate(num_latent_gps):
        for k, seed in enumerate(seeds):
            torch.random.manual_seed(seed)
            model = DeepLMC(10,
                    num_tasks=2*129,
                    num_hidden_dgp_dims=hidden_dim,
                    num_latent_gps=num_latent,
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
                logfile=f"dgp_{hidden_dim}_{num_latent}_{lr}_{seed}".replace(".", "") + ".log",
                savefile=f"dgp_{hidden_dim}_{num_latent}_{lr}_{seed}".replace(".", "") + ".model"
                )

