import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import torch
from torch.utils.data import TensorDataset, DataLoader
import sbi.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import os, itertools, pickle, h5py
from scipy.signal import welch
from functools import partial
import train_likelihood

BATCH_SIZE = 100

device = "cuda"
torch.random.manual_seed(123)

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

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

hidden_features = [258, 400, 600, 800, 1000, 2000, 4000, 6000, 8000]
num_transforms = [2, 4, 6, 8, 10]
lrs = [0.01, 0.001, 0.0001]
def build_maf(num_hidden_features, num_transforms):
    transforms = []
    for i in range(num_transforms):
        if i != 0:
            transforms.append(T.Permute(torch.randperm(258, dtype=torch.int64)).inv.to(device))
        transforms.append(T.ConditionalAffineAutoregressive(
            pyro.nn.ConditionalAutoRegressiveNN(258,
                                                10,
                                                [num_hidden_features],
                                                skip_connections=True)).inv.to(device))

        base_dist = dist.Normal(torch.zeros(258).to(device), torch.ones(258).to(device))
        flow = dist.ConditionalTransformedDistribution(base_dist, transforms)
        return flow

def train_batch_maf(x_batch, y_batch, flow, opt):
    opt.zero_grad()
    log_prob = flow.condition(x_batch).log_prob(y_batch)
    loss = -log_prob.mean()
    loss.backward()
    opt.step()
    return loss.item() 

def eval_batch_maf(x_batch, y_batch, flow):
    log_prob = flow.condition(x_batch).log_prob(y_batch)
    loss = -log_prob.mean()
    return loss.item() 
 
for i, _ in enumerate(lrs):
    for j, _ in enumerate(hidden_features):
        for k, _ in enumerate(num_transforms):
            logfile = os.path.join(".", "error_log_maf.csv")

            flow = build_maf(hidden_features[j], num_transforms[k])
    
            trainable_params = itertools.chain(*[t.parameters() for t in flow.transforms if hasattr(t, "parameters")])
            opt = torch.optim.Adam(trainable_params, lr=lrs[i])
   
            train_batch_maf_partial = partial(train_batch_maf, flow=flow, opt=opt)
            eval_batch_maf_partial = partial(eval_batch_maf, flow=flow)
    
            train_likelihood.STOP_PERSISTENCE = 4 
            train_likelihood.EVALUATION_FREQUENCY = 5
            train_likelihood.train(
                flow,
                train_batch_maf_partial,
                eval_batch_maf_partial,
                train_loader,
                test_loader,
                logfile=f"maf_{hidden_features[j]}_{num_transforms[k]}_{lrs[i]}".replace(".", "") + ".log",
                savefile=f"maf_{hidden_features[j]}_{num_transforms[k]}_{lrs[i]}".replace(".", "") + ".model")
    
