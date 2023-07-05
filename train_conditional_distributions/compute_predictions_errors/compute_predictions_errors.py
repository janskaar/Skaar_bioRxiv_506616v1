import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
from deep_lmc_natural_gradients import DeepLMC
import gpytorch, torch
from torch.utils.data import TensorDataset, DataLoader
import os, time, pickle, h5py
from scipy.signal import welch
import numpy as np

EVAL_BATCH_SIZE = 20
EVAL_BATCH_SIZE_LFP = 5
device = "cuda"
torch.random.manual_seed(1234)

savedir = "."

n_train = 8000
dgp_savefile = os.path.join("dgp_20_50_0001_13.model")
dgp_lfp_savefile = os.path.join("dgp_20_50_lfp.model")
maf_savefile = os.path.join("maf_1000_8_0001.model")

with h5py.File('lfp_kernel_delta.h5', 'r') as f:
    ex_kernel_delta = f['ex'][()]
    in_kernel_delta = f['in'][()]

# Alpha function for computing LFPs
def isyn_fn(J, tau):
    t = np.arange(0, 100, 0.1)
    return - J * t/tau * np.exp(1 - t/tau) * 1e-3

n_train = 8000

sim_path = os.path.join("../../simulation_scripts/brunel_simulations_10p")
dirs = sorted(os.listdir(os.path.join(sim_path, "nest_output")))


params = []
hists = []
lfps = []
# loop through directories containing individual simulations
for i, d in enumerate(dirs[:5]):
    with open(os.path.join(sim_path, "parameters", d + ".pkl"), "rb") as f:
        param = pickle.load(f)
        params.append(np.array([param["eta"], param["g"], param["tauSyn"], param["tauMem"], param["delay"], param["t_ref"], param["cSyn"], param["theta"], param["CMem"], param["V_reset"]]))

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
            ex_hist = f["ex_hist"][()][501:]
            in_hist = f["in_hist"][()][501:]
            lfp = []
            for j in range(6):
                lfp.append(np.convolve(ex_hist, ex_kernel[j], mode='valid') + np.convolve(in_kernel[j], in_hist, mode='valid'))
            lfp = np.array(lfp)
            lfp -= lfp.mean(1, keepdims=True)
            lfps.append(lfp)
            hists.append(np.stack((ex_hist, in_hist)))

hists = np.array(hists)
lfps = np.array(lfps)
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

# LFP
fs, lfp_psds = welch(lfps[~args,:,501:])
lfp_psds = lfp_psds.reshape((-1, 129*6))

Y_lfp = torch.tensor(np.log10(lfp_psds.copy().astype(np.float32)))
Ytrain_lfp_mean = Y_lfp[:n_train].mean(0, keepdims=True)
Yn_lfp = Y_lfp - Ytrain_lfp_mean

train_y_lfp = torch.Tensor(Yn_lfp[:n_train]).to(device=device)
test_y_lfp = torch.Tensor(Yn_lfp[n_train:]).to(device=device)

print(train_x.shape, flush=True)
print(train_y.shape, flush=True)
print(test_x.shape, flush=True)
print(test_y.shape, flush=True)

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

eval_test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
eval_train_loader = DataLoader(train_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

# LFP
train_dataset_lfp = TensorDataset(train_x, train_y_lfp)
test_dataset_lfp = TensorDataset(test_x, test_y_lfp)

eval_test_loader_lfp = DataLoader(test_dataset_lfp, batch_size=EVAL_BATCH_SIZE_LFP, shuffle=False)
eval_train_loader_lfp = DataLoader(train_dataset_lfp, batch_size=EVAL_BATCH_SIZE_LFP, shuffle=False)

num_tasks = train_y.size(-1)
num_inducing1 = 128
num_inducing2 = 128

def eval_batch_dgp(x_batch, y_batch, model):
    # reshape to (num, 1, output_features), so we
    # compute distribution independently for each example
    dist = model.likelihood(model.memory_efficient_call(x_batch[:,None,:]))
    log_probs = dist.log_prob(y_batch[None,:,None,:])
    log_probs = torch.logsumexp(log_probs, 0) - torch.log(torch.tensor(log_probs.shape[0]).to(device))
    distmean = dist.mean.squeeze()
    distvariance = dist.variance.squeeze()
    mean = distmean.mean(0)
    variance = torch.mean(distmean**2 + distvariance, axis=0) - distmean.mean(axis=0)**2
    error = mean.squeeze() - y_batch
    return -log_probs, error, mean, variance


##################################################
# Population spiking
##################################################


model = DeepLMC(10,
    num_tasks=2*129,
    num_hidden_dgp_dims=20,
    num_latent_gps=50,
    num_inducing1=num_inducing1,
    num_inducing2=num_inducing2).to(device)

state_dict = torch.load(dgp_savefile) 
model.load_state_dict(state_dict)

with torch.no_grad():
    test_loss = []
    test_mean = []
    test_error = []
    test_var = []
    for x_batch, y_batch in eval_test_loader:
        loss, error, mean, var = eval_batch_dgp(x_batch, y_batch, model)
        if len(error.shape) == 1:
            error = error[None,:]
        if len(mean.shape) == 1:
            mean = mean[None,:]
        if len(var.shape) == 1:
            var = var[None,:]
        test_loss.append(loss.cpu().numpy())
        test_mean.append(mean.cpu().numpy())
        test_error.append(error.cpu().numpy())
        test_var.append(var.cpu().numpy())
    test_loss = np.concatenate(test_loss)
    test_error = np.concatenate(test_error)
    test_mean = np.concatenate(test_mean)
    test_var = np.concatenate(test_var)

    train_loss = []
    train_mean = []
    train_error = []
    train_var = []
    for x_batch, y_batch in eval_train_loader:
        loss, error, mean, var = eval_batch_dgp(x_batch, y_batch, model)
        if len(error.shape) == 1:
            error = error[None,:]
        if len(mean.shape) == 1:
            mean = mean[None,:]
        if len(var.shape) == 1:
            var = var[None,:]
 
        train_loss.append(loss.cpu().numpy())
        train_mean.append(mean.cpu().numpy())
        train_error.append(error.cpu().numpy())
        train_var.append(var.cpu().numpy())
    train_loss = np.concatenate(train_loss)
    train_error = np.concatenate(train_error)
    train_mean = np.concatenate(train_mean)
    train_var = np.concatenate(train_var)

np.save(os.path.join(savedir, "dgp_train_loss.npy"), train_loss)
np.save(os.path.join(savedir, "dgp_test_loss.npy"), test_loss)
np.save(os.path.join(savedir, "dgp_train_error.npy"), train_error)
np.save(os.path.join(savedir, "dgp_train_mean.npy"), train_mean)
np.save(os.path.join(savedir, "dgp_train_var.npy"), train_var)
np.save(os.path.join(savedir, "dgp_test_error.npy"), test_error)
np.save(os.path.join(savedir, "dgp_test_mean.npy"), test_mean)
np.save(os.path.join(savedir, "dgp_test_var.npy"), test_var)

##################################################
# LFP
##################################################

model = DeepLMC(10,
    num_tasks=6*129,
    num_hidden_dgp_dims=20,
    num_latent_gps=50,
    num_inducing1=num_inducing1,
    num_inducing2=num_inducing2).to(device)

state_dict = torch.load(dgp_lfp_savefile) 
model.load_state_dict(state_dict)

with torch.no_grad():
    test_loss = []
    test_mean = []
    test_error = []
    test_var = []
    for x_batch, y_batch in eval_test_loader_lfp:
        loss, error, mean, var = eval_batch_dgp(x_batch, y_batch, model)
        if len(error.shape) == 1:
            error = error[None,:]
        if len(mean.shape) == 1:
            mean = mean[None,:]
        if len(var.shape) == 1:
            var = var[None,:]
        test_loss.append(loss.cpu().numpy())
        test_mean.append(mean.cpu().numpy())
        test_error.append(error.cpu().numpy())
        test_var.append(var.cpu().numpy())
    test_loss = np.concatenate(test_loss)
    test_error = np.concatenate(test_error)
    test_mean = np.concatenate(test_mean)
    test_var = np.concatenate(test_var)

    train_loss = []
    train_mean = []
    train_error = []
    train_var = []
    for x_batch, y_batch in eval_train_loader_lfp:
        loss, error, mean, var = eval_batch_dgp(x_batch, y_batch, model)
        if len(error.shape) == 1:
            error = error[None,:]
        if len(mean.shape) == 1:
            mean = mean[None,:]
        if len(var.shape) == 1:
            var = var[None,:]
 
        train_loss.append(loss.cpu().numpy())
        train_mean.append(mean.cpu().numpy())
        train_error.append(error.cpu().numpy())
        train_var.append(var.cpu().numpy())
    train_loss = np.concatenate(train_loss)
    train_error = np.concatenate(train_error)
    train_mean = np.concatenate(train_mean)
    train_var = np.concatenate(train_var)

np.save(os.path.join(savedir, "dgp_lfp_train_loss.npy"), train_loss)
np.save(os.path.join(savedir, "dgp_lfp_test_loss.npy"), test_loss)
np.save(os.path.join(savedir, "dgp_lfp_train_error.npy"), train_error)
np.save(os.path.join(savedir, "dgp_lfp_train_mean.npy"), train_mean)
np.save(os.path.join(savedir, "dgp_lfp_train_var.npy"), train_var)
np.save(os.path.join(savedir, "dgp_lfp_test_error.npy"), test_error)
np.save(os.path.join(savedir, "dgp_lfp_test_mean.npy"), test_mean)
np.save(os.path.join(savedir, "dgp_lfp_test_var.npy"), test_var)


############################################################
# MAF
############################################################


maf = torch.load(maf_savefile)

def eval_batch_maf(x_batch, y_batch, flow):
    conditioned = flow.condition(x_batch)
    log_prob = conditioned.log_prob(y_batch)
    loss = -log_prob
    samples = conditioned.sample((len(x_batch),))
    mean = samples
    errors = y_batch - mean
    return loss, errors, mean 
 
with torch.no_grad():
    test_loss = []
    test_mean = []
    test_error = []
    test_var = []
    for x_batch, y_batch in eval_test_loader:
        loss, error, mean = eval_batch_maf(x_batch, y_batch, maf)
        if len(error.shape) == 1:
            error = error[None,:]
        if len(mean.shape) == 1:
            mean = mean[None,:]
        if len(var.shape) == 1:
            var = var[None,:]
 
        test_loss.append(loss.cpu().numpy())
        test_mean.append(mean.cpu().numpy())
        test_error.append(error.cpu().numpy())
        test_var.append(var.cpu().numpy())
    test_loss = np.concatenate(test_loss)
    test_error = np.concatenate(test_error)
    test_mean = np.concatenate(test_mean)
    test_var = np.concatenate(test_var)



    train_loss = []
    train_mean = []
    train_error = []
    for x_batch, y_batch in eval_train_loader:
        loss, error, mean = eval_batch_maf(x_batch, y_batch, maf)
        if len(error.shape) == 1:
            error = error[None,:]
        if len(mean.shape) == 1:
            mean = mean[None,:]
        if len(var.shape) == 1:
            var = var[None,:]
 
        train_loss.append(loss.cpu().numpy())
        train_mean.append(mean.cpu().numpy())
        train_error.append(error.cpu().numpy())
    train_loss = np.concatenate(train_loss)
    train_error = np.concatenate(train_error)
    train_mean = np.concatenate(train_mean)


print(train_loss.shape)
print(train_error.shape)
print(train_mean.shape)
print(train_var.shape)

print(test_loss.shape)
print(test_error.shape)
print(test_mean.shape)
print(test_var.shape)

np.save(os.path.join(savedir, "maf_train_loss.npy"), train_loss)
np.save(os.path.join(savedir, "maf_test_loss.npy"), test_loss)
np.save(os.path.join(savedir, "maf_train_error.npy"), train_error)
np.save(os.path.join(savedir, "maf_train_mean.npy"), train_mean)
np.save(os.path.join(savedir, "maf_train_var.npy"), train_var)
np.save(os.path.join(savedir, "maf_test_error.npy"), test_error)
np.save(os.path.join(savedir, "maf_test_mean.npy"), test_mean)
np.save(os.path.join(savedir, "maf_test_var.npy"), test_var)

