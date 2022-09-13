import numpy as np
import os, time
import matplotlib.pyplot as plt
import matplotlib.colors
import copy
import gpytorch
import torch
from sklearn.neighbors import KernelDensity

from deep_lmc_natural_gradients import DeepLMC

rank = int(os.environ['SLURM_PROCID'])
size = int(os.environ['SLURM_NTASKS'])
torch.random.manual_seed(123)
np.random.seed(1234)
#size = 1
#rank = 0

n_train = 3000
dpath = os.path.join('../../simulation_scripts/')

psds = np.load(os.path.join(dpath, '10p_hist_psd_nosync.npy'))
labels = np.load(os.path.join(dpath, '10p_labels_nosync.npy')).astype(np.float64)

psds = np.concatenate([psds[:,0,:], psds[:,1,:]], axis=1)

minlabels = labels.min(axis=0)
difflabels = labels.max(axis=0) - labels.min(axis=0)
labels -= minlabels
labels /= difflabels

X = labels.copy()
Y = np.log10(psds.copy())
Yn = Y - Y.mean(axis=0, keepdims=True)

train_x = torch.Tensor(X[:n_train])
train_y = torch.Tensor(Yn[:n_train])

test_x = torch.Tensor(X[3000:])
test_y = torch.Tensor(Yn[3000:])

num_inducing1 = 128
num_inducing2 = 128
model = DeepLMC(labels.shape,
                num_latent_gps=14,
                num_hidden_dgp_dims=10,
                num_tasks=258,
                num_inducing1=num_inducing1,
                num_inducing2=num_inducing2
                )

state_dict = torch.load('../../training_scripts/deep_lmc_10p_hist_nosync/deep_lmc_4.pth')
model.load_state_dict(state_dict)
model.eval();

class MH:
    def __init__(self, model, y0, cov = None, sigma = 0.5, num_chains=5):
        self.model = model
        self.num_chains = num_chains
        self.cov0 = torch.diag(torch.ones(10)).tile((self.num_chains, 1, 1)) * 1e-6 # initial covariance
        self.cov = torch.zeros((self.num_chains, 10, 10))                           # sample covariance to be updated
        self.loc = torch.zeros((self.num_chains, 10))                               # zero mean for proposal

        self.zero_loc = torch.zeros((self.num_chains, 10))
        self.num_accepted = np.zeros(self.num_chains, dtype=np.float32)
        self.sigma = torch.ones(self.num_chains) * sigma
        self.y0 = y0
        self.proposal = torch.distributions.MultivariateNormal(self.zero_loc, \
            torch.diag(torch.ones(10) * 0.1**2).tile((self.num_chains, 1, 1)) + self.cov0)
        self.ma = np.zeros(self.num_chains, dtype=np.float32)
        self.ma_factor = 0.02

    def step(self, pos):
        ## Evaluate current log probability again as it is a monte carlo estimate,
        ## to avaid getting stuck in 'lucky' draws
        current_log_p = self.model.batch_data_independent_log_prob(pos, y0)

        ## Move
        new_pos = pos + self.proposal.sample()

        ## Check if any positions outside prior
        outside = (new_pos > 1.).any(axis=1) | (new_pos < 0.).any(axis=1)
        new_log_p = self.model.batch_data_independent_log_prob(new_pos, y0)

        ## Compute ratio and correct for prior
        a = torch.exp(new_log_p - current_log_p)
        a[outside] = -1.

        ## Accept with probabilities a < u ~ U(0,1)
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

            ## Update covariance every 10 steps
            if i % 10 == 0 and i > 20:
                self.update_covariance()

            ## Adjust sigma
            if i > 20:
                self.sigma[self.ma > 0.2] *= 1.01
                self.sigma[self.ma < 0.2] *= 0.99
                sigma_ctr = 0

    def update_covariance(self):
        ## Compute covariance of samples collected so far
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
        ## Get accepted positions
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

        ## Find chains with roughly same log prob as the best
        good_chains = mean_log_probs > highest_log_prob - 1.

        ## If there are multiple, use chain with highest marginal
        ## standard deviations, averaged across parameters
        if good_chains.sum() > 1:
            maxarg = mean_stds[good_chains].argmax()
        else:
            maxarg = 0
        use_chain = np.arange(self.num_chains)[good_chains][maxarg]

        ## Keep index for finding initial position for sampling
        self.use_chain = use_chain

        ## set up proposal distribution for mh run
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

num_burn_in = 12000
num_steps = 28000
num_chains = 5
ind = rank
savedir = os.path.join('.', f'{ind:04d}')
os.mkdir(savedir)
print('STARTING INDEX ', ind, flush=True)
y0 = test_y[ind:ind+1]
x0 = test_x[ind:ind+1]

with torch.no_grad():
    pos_init = torch.rand((num_chains, 10))
    mh =  MH(model, y0, num_chains=num_chains)
    print('STARTING BURN-IN')
    mh.burn_in(pos_init, num_burn_in)
    np.save(os.path.join(savedir, 'burnin_log_probs.npy'), mh.burnin_log_probs)
    np.save(os.path.join(savedir, 'burnin_acceptance.npy'), mh.burnin_acceptance)
    np.save(os.path.join(savedir, 'burnin_proposals.npy'), mh.burnin_proposals)
    np.save(os.path.join(savedir, 'sigma.npy'), np.array(mh.sigma))
    np.save(os.path.join(savedir, 'covariance.npy'), mh.cov.numpy())

    mh.find_best_chain()

    print('Starting sampling')
    mh.run_mh(num_steps)
    np.save(os.path.join(savedir, 'log_probs.npy'), mh.log_probs)
    np.save(os.path.join(savedir, 'acceptance.npy'), mh.acceptance)
    np.save(os.path.join(savedir, 'proposals.npy'), mh.proposals)
