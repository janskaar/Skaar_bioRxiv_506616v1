import torch
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import NaturalVariationalDistribution, TrilNaturalVariationalDistribution, VariationalStrategy, LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.models import ApproximateGP
from gpytorch.likelihoods import MultitaskGaussianLikelihood
import numpy as np
from gpytorch.means import Mean, ConstantMean

class SkipMean(Mean):
    def __init__(self, input_size, output_size, batch_shape=torch.Size()):
        super().__init__()
        self.W = torch.zeros(output_size, input_size)
        self.W[:input_size,:input_size] = torch.eye(input_size)

    def forward(self, x):
        m = torch.sum(x.transpose(-3, -2) * self.W, axis=-1).transpose(-1,-2)
        return m

class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, skip_mean=True):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])

        variational_distribution = TrilNaturalVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape,
            mean_init_std=0.
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = ConstantMean() if not skip_mean else SkipMean(input_dims, output_dims)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class DGPLCMLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, latent_dims, num_inducing=128, skip_mean=True):
        inducing_points = torch.randn(latent_dims, num_inducing, input_dims)
        batch_shape = torch.Size([latent_dims])

        variational_distribution = TrilNaturalVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        base_variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        variational_strategy = LMCVariationalStrategy(
            base_variational_strategy,
            num_tasks=output_dims,
            num_latents=latent_dims,
            latent_dim=-1)

        super().__init__(variational_strategy, input_dims, latent_dims)
        self.mean_module = ConstantMean() if not skip_mean else SkipMean(input_dims)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, inputs, are_samples=False, **kwargs):
        """
        Slightly adapted from original GPyTorch code to allow LMC in deep layer
        """
        deterministic_inputs = not are_samples
        if isinstance(inputs, MultitaskMultivariateNormal):
            inputs = torch.distributions.Normal(loc=inputs.mean, scale=inputs.variance.sqrt()).rsample()
            deterministic_inputs = False

        if gpytorch.settings.debug.on():
            if not torch.is_tensor(inputs):
                raise ValueError(
                    "`inputs` should either be a MultitaskMultivariateNormal or a Tensor, got "
                    f"{inputs.__class__.__Name__}"
                )

            if inputs.size(-1) != self.input_dims:
                raise RuntimeError(
                    f"Input shape did not match self.input_dims. Got total feature dims [{inputs.size(-1)}],"
                    f" expected [{self.input_dims}]"
                )

        # Repeat the input for all possible outputs
        if self.output_dims is not None:
            inputs = inputs.unsqueeze(-3)
            inputs = inputs.expand(*inputs.shape[:-3], self.output_dims, *inputs.shape[-2:])


        # Now run samples through the GP
        output = ApproximateGP.__call__(self, inputs)

        ###############################################
        # Removed block diagonalization of batched GPs
        ###############################################
        return output

    def data_independent_call(self, inputs, are_samples=False, likelihood=None, **kwargs):
        """
        LMC call for data independent distribution (non-lazy), modified
        from standard LMC variational strategy in GPyTorch.
        """
        inputs = torch.distributions.Normal(loc=inputs.mean, scale=inputs.variance.sqrt()).rsample()

        # Repeat the input for all possible outputs
        inputs = inputs.unsqueeze(-3)
        inputs = inputs.expand(*inputs.shape[:-3], self.output_dims, *inputs.shape[-2:])

        ### Begin LMCVariationalStrategy call
        function_dist = self.variational_strategy.base_variational_strategy(inputs, **kwargs)
        lmc_coefficients = self.variational_strategy.lmc_coefficients.expand(*function_dist.batch_shape, self.variational_strategy.lmc_coefficients.size(-1))
        num_batch = len(function_dist.batch_shape)
        num_dim = num_batch + len(function_dist.event_shape)
        latent_dim = num_batch + self.variational_strategy.latent_dim if self.variational_strategy.latent_dim is not None else None

        # Mean
        mean = function_dist.mean.permute(*range(0, latent_dim), *range(latent_dim + 1, num_dim), latent_dim)
        mean = mean @ lmc_coefficients.permute(
            *range(0, latent_dim), *range(latent_dim + 1, num_dim - 1), latent_dim, -1
        )

        # Covar
        covar = function_dist.covariance_matrix
        lmc_factor = torch.matmul(lmc_coefficients.unsqueeze(-1), lmc_coefficients.unsqueeze(-2))

        diag = torch.diagonal(covar, dim1=-1, dim2=-2)
        covar = diag[...,None,None] * lmc_factor[...,None,:,:] #   dims (N_likelihood_samples, N_latent_gps, num_data, 1, 1)
                                                               # x dims (N_likelihood_samples, N_latent_gps, 1, output_dim, output_dim)
                                                               #-> dims (N_likelihood_samples, N_latent_gps, num_data, output_dim, output_dim)

        covar = torch.sum(covar, dim=latent_dim)               #-> dims (N_likelihood_samples, num_data, output_dim, output_dim)


        # Add a bit of jitter to make the covar PD
        covar = covar + torch.eye(covar.size(-1))[None,None,...] * 1e-6

        if likelihood is not None:
            covar = covar + torch.diag(likelihood.task_noises + likelihood.noise)

        # Done!
        function_dist = MultivariateNormal(mean, covar)        # Independent Multivariate Normals w batch shape (N_likelihood_samples, num_data)
                                                               # and event shape (n_outputs)
        return function_dist

class DeepLMC(DeepGP):
    def __init__(self, train_x_shape, num_tasks=129, num_latent_gps=12, num_hidden_dgp_dims=5, num_inducing1=128, num_inducing2=128):
        hidden_layer1 = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dgp_dims,
            skip_mean=True,
            num_inducing=num_inducing1
        )

        last_layer = DGPLCMLayer(
            input_dims= hidden_layer1.output_dims,
            output_dims= num_tasks,
            latent_dims= num_latent_gps,
            num_inducing=num_inducing2,
            skip_mean= False
        )

        super().__init__()

        self.hidden_layer1 = hidden_layer1
        self.last_layer = last_layer

        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks, rank=0)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer1(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    def data_independent_dist(self, inputs, likelihood=False):
        hidden_rep1 = self.hidden_layer1(inputs)
        output = self.last_layer.data_independent_call(hidden_rep1, likelihood=self.likelihood if likelihood else None)
        return output

    def batch_mean_variance(self, inputs, likelihood=False, batch_size=200):
        with torch.no_grad():
            num_batches = int(len(inputs) / batch_size) if len(inputs) % batch_size == 0 else int(len(inputs) / batch_size) + 1
            means = []
            variances = []
            for i in range(num_batches):
                dist = self.data_independent_dist(inputs[i*batch_size:(i+1)*batch_size], likelihood=likelihood)
                means.append(dist.mean.mean(0))
                variance = torch.mean(dist.mean**2 + dist.variance, axis=0) - dist.mean.mean(axis=0)**2
                variances.append(variance)
            return torch.cat(means, axis=0), torch.cat(variances, axis=0)

    def batch_data_independent_log_prob(self, inputs, y, batch_size=200, likelihood=True):
        with torch.no_grad():
            num_batches = int(len(inputs) / batch_size) if len(inputs) % batch_size == 0 else int(len(inputs) / batch_size) + 1
            logprobs = []
            for i in range(num_batches):
                dist = self.data_independent_dist(inputs[i*batch_size:(i+1)*batch_size], likelihood=likelihood)
                logprob = dist.log_prob(y)
                logprob = torch.logsumexp(logprob, 0) - torch.log(torch.tensor(logprob.shape[0]))
                logprobs.append(logprob)
            return torch.cat(logprobs)
