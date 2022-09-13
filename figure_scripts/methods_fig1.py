import torch
import gpytorch
import matplotlib.pyplot as plt
from gpytorch.kernels import ScaleKernel, RBFKernel

TICK_FONT_SIZE = 7
TITLE_FONT_SIZE = 7
LABEL_FONT_SIZE = 7

torch.manual_seed(1234)

kernel1 = ScaleKernel(
            RBFKernel(ard_num_dims=1), ard_num_dims=None)
kernel1.base_kernel.lengthscale = 0.4
kernel2 = ScaleKernel(
            RBFKernel(ard_num_dims=1), ard_num_dims=None)
kernel2.base_kernel.lengthscale = 0.3

N = 100
x = torch.linspace(0,1,N)
mean = torch.zeros(N)

samples = [x.unsqueeze(0).tile((5,1))]
for i in range(5):
    sample = []
    for j in range(5):
        cov = kernel1(samples[-1][j])
        mvn = gpytorch.distributions.MultivariateNormal(mean, cov)
        s = mvn.sample()
        sample.append(s)
    samples.append(torch.stack(sample))
# mean2 = torch.zeros(N)
# samples2 = []
# for i in range(5):
#     cov2 = kernel2(samples1[i])
#     mvn2 = gpytorch.distributions.MultivariateNormal(mean2, cov2)
#     samples2.append(mvn2.sample(torch.Size([1])).squeeze())
#
# mean3 = torch.zeros(N)
# samples3 = []
# for i in range(5):
#     cov3 = kernel2(samples2[i])
#     mvn3 = gpytorch.distributions.MultivariateNormal(mean3, cov3)
#     samples3.append(mvn2.sample(torch.Size([1])).squeeze())

samples4 = samples[-2].numpy()
samples2 = samples[2].numpy()
samples1 = samples[1].numpy()

fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True)
fig.set_size_inches([7, 1.3])
for i in range(5):
    ax[0].plot(x, samples1[i])
    ax[1].plot(x, samples2[i])
    ax[2].plot(x, samples4[i])
labels = ['layer 1', 'layer 2', 'layer 4']
for i in range(3):
    ax[i].set_xlabel('x',  fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=0.)
    ax[i].set_title(labels[i], fontsize=LABEL_FONT_SIZE)
    ax[i].tick_params(which='both', labelsize=TICK_FONT_SIZE)
ax[0].set_ylabel('f(x)',  fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=0.)
fig.savefig('methods_fig1.pdf', bbox_inches='tight')
plt.show()
