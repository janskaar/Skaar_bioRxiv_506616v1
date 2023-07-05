import numpy as np
import os, time, pickle, h5py
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.signal import welch
from functools import partial
import gpytorch
import torch

from deep_lmc_natural_gradients import DeepLMC

torch.random.manual_seed(123)
np.random.seed(1234)
DEVICE = "cpu"
print(torch.cuda.is_available())
print(torch.cuda.device_count())
n_train = 8000

sim_path = os.path.join("../simulation_scripts/brunel_simulations_10p")
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

train_x = torch.Tensor(X[:n_train]).to(device=DEVICE)
train_y = torch.Tensor(Yn[:n_train]).to(device=DEVICE)

test_x = torch.Tensor(X[n_train:]).to(device=DEVICE)
test_y = torch.Tensor(Yn[n_train:]).to(device=DEVICE)

print(f"train_x shape: {train_x.shape}", flush=True)
print(f"test_x shape: {test_x.shape}", flush=True)
print(f"train_y shape: {train_y.shape}", flush=True)
print(f"test_y shape: {test_y.shape}", flush=True)

class MH:
    def __init__(self, eval_likelihood, y0, cov = None, sigma = 0.5, num_chains=5):
        self.eval_likelihood = eval_likelihood
        self.num_chains = num_chains
        self.cov0 = torch.diag(torch.ones(10)).tile((self.num_chains, 1, 1)) * 1e-6
        self.cov = torch.zeros((self.num_chains, 10, 10))
        self.loc = torch.zeros((self.num_chains, 10))

        self.zero_loc = torch.zeros((self.num_chains, 10))
        self.num_accepted = np.zeros(self.num_chains, dtype=np.float32)
        self.sigma = torch.ones(self.num_chains) * sigma
        self.y0 = y0
        self.proposal = torch.distributions.MultivariateNormal(self.zero_loc, \
            torch.diag(torch.ones(10) * 0.1**2).tile((self.num_chains, 1, 1)) + self.cov0)
        self.ma = np.zeros(self.num_chains, dtype=np.float32)
        self.ma_factor = 0.02

    def step(self, pos):
        current_log_p = self.eval_likelihood(pos, y0)

        # Move
        new_pos = pos + self.proposal.sample()

        # Check if any positions outside prior
        outside = (new_pos > 1.).any(axis=1) | (new_pos < 0.).any(axis=1)
        new_log_p = self.eval_likelihood(new_pos, y0)

        # Compute ratio and correct for prior
        a = torch.exp(new_log_p - current_log_p)
        a[outside] = -1.

        # Accept with probabilities a < u ~ U(0,1)
        accept_prob = torch.rand(pos.shape[0])
        accepted = accept_prob < a
        pos[accepted] = new_pos[accepted]

        return new_pos, pos, new_log_p, accepted

    def burn_in(self, pos, num_steps):
        sigma_ctr = 0
        self.burnin_proposals = np.zeros((num_steps, self.num_chains, 10), dtype=np.float32)
        self.burnin_log_probs = np.zeros((num_steps, self.num_chains), dtype=np.float32)
        self.burnin_acceptance = np.zeros((num_steps, self.num_chains), dtype=bool)
        for i in range(num_steps):
            if i % 100 == 0:
                print('STEP ', i, flush=True)
                print('AVG ACCEPTANCE RATES: ', self.ma, flush=True)
            proposal, pos, log_p, accepted = self.step(pos)
            accepted = accepted.numpy()
            self.num_accepted += accepted
            self.burnin_proposals[i] = proposal
            self.burnin_acceptance[i] = accepted
            self.burnin_log_probs[i] = log_p

            # update acceptance moving average
            self.ma *= 1 - self.ma_factor
            self.ma += self.ma_factor * accepted

            # Update covariance every 10 steps
            if i % 10 == 0 and i > 20:
                self.update_covariance()

            # Adjust sigma
            if i > 20:
                self.sigma[self.ma > 0.2] *= 1.01
                self.sigma[self.ma < 0.2] *= 0.99
                sigma_ctr = 0

    def update_covariance(self):
        covs = []
        for i in range(self.num_chains):
            s = self.burnin_proposals[:,i][self.burnin_acceptance[:,i]].copy()
            if len(s) < 2:
                c = self.cov0[0]
            else:
                s = s[-1000:]
                s -= s.mean(0, keepdims=True)
                c = torch.tensor(s.T.dot(s) / len(s))
            covs.append(c)
        self.cov = torch.stack(covs, axis=0)
        self.proposal = torch.distributions.MultivariateNormal(self.zero_loc, self.sigma[:,None,None] * self.cov + self.cov0)

    def find_best_chain(self):
        # Get accepted positions
        mean_log_probs = []
        mean_stds = []
        for i in range(self.num_chains):
            log_prob = self.burnin_log_probs[:,i][self.burnin_acceptance[:,i]]
            sample = self.burnin_proposals[:,i][self.burnin_acceptance[:,i]][-1000:].copy()
            mean_std = sample.std(axis=0).mean()
            if log_prob.shape[0] == 0:
                mean = 0.
            else:
                mean = log_prob[-1000:].mean()
            mean_log_probs.append(mean)
            mean_stds.append(mean_std)
        mean_stds = np.array(mean_stds)
        mean_log_probs = np.array(mean_log_probs)
        highest_log_prob = mean_log_probs.max()

        # Find chains with roughly same log prob as the best
        good_chains = mean_log_probs > highest_log_prob - 1.

        # If there are multiple, use chain with highest marginal
        # standard deviations, averaged across parameters
        if good_chains.sum() > 1:
            maxarg = mean_stds[good_chains].argmax()
        else:
            maxarg = 0
        use_chain = np.arange(self.num_chains)[good_chains][maxarg]

        # Keep index for finding initial position for sampling
        self.use_chain = use_chain

        # set up proposal distribution for mh run
        use_cov = torch.stack([self.cov[use_chain]]*self.num_chains, axis=0)
        use_sigma = self.sigma[use_chain]

        self.proposal = torch.distributions.MultivariateNormal(self.loc, use_sigma * use_cov + self.cov0)

    def get_init_pos(self):
        init_pos = self.burnin_proposals[:,self.use_chain,:][self.burnin_acceptance[:,self.use_chain]][-1]
        init_pos = np.stack([init_pos]*self.num_chains, axis=0)
        return torch.tensor(init_pos)

    def run_mh(self, num_steps):
        self.log_probs = np.zeros((num_steps, self.num_chains), dtype=np.float32)
        self.proposals = np.zeros((num_steps, self.num_chains, 10), dtype=np.float32)
        self.acceptance = np.zeros((num_steps, self.num_chains), dtype=bool)

        pos = self.get_init_pos()
        for i in range(num_steps):
            if i % 100 == 0:
                print('STEP ', i, flush=True)
            proposal, pos, log_p, accepted = self.step(pos)
            self.proposals[i] = proposal
            self.acceptance[i] = accepted
            self.log_probs[i] = log_p

def eval_maf_likelihood(sample, y0, model):
    num = len(sample)
    return model.condition(sample).log_prob(y0.tile((num,1)))


maf_likelihood = torch.load("maf_1000_8_0001.model", map_location=torch.device(DEVICE))

eval_maf_likelihood = partial(eval_maf_likelihood, model=maf_likelihood)

num_burn_in = 12000
num_steps = 28000
num_chains = 10
for ind in range(100):
    savedir = os.path.join('mcmc_posterior_samples_maf', f'{ind:04d}')
    os.mkdir(savedir)
    print('STARTING INDEX ', ind, flush=True)
    y0 = test_y[ind:ind+1]
    x0 = test_x[ind:ind+1]
    
    with torch.no_grad():
        pos_init = torch.rand((num_chains, 10))
        mh =  MH(eval_maf_likelihood, y0, num_chains=num_chains)
        print('STARTING BURN-IN', flush=True)
        tic = time.perf_counter()
        mh.burn_in(pos_init, num_burn_in)
        toc = time.perf_counter()
        print(f"Burn-in complete: {num_burn_in} steps in {toc - tic} seconds", flush=True)
        np.save(os.path.join(savedir, 'burnin_log_probs.npy'), mh.burnin_log_probs)
        np.save(os.path.join(savedir, 'burnin_acceptance.npy'), mh.burnin_acceptance)
        np.save(os.path.join(savedir, 'burnin_proposals.npy'), mh.burnin_proposals)
        np.save(os.path.join(savedir, 'sigma.npy'), np.array(mh.sigma))
        np.save(os.path.join(savedir, 'covariance.npy'), mh.cov.numpy())
    
        mh.find_best_chain()
    
        print('Starting sampling', flush=True)
        tic = time.perf_counter()
        mh.run_mh(num_steps)
        toc = time.perf_counter()
        print(f"Sampling complete: {num_steps} steps in {toc - tic} seconds", flush=True)
    
        np.save(os.path.join(savedir, 'log_probs.npy'), mh.log_probs)
        np.save(os.path.join(savedir, 'acceptance.npy'), mh.acceptance)
        np.save(os.path.join(savedir, 'proposals.npy'), mh.proposals)

