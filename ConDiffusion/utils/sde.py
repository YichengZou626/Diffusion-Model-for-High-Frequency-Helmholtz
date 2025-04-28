"""Score modules"""

import math
import torch
import torch.nn as nn
import jax.numpy as jnp
import jax
from torch import Size, Tensor
from tqdm import tqdm
from typing import *
from zuko.utils import broadcast

def get_time_category(t, boundary_ranges: tuple = (1.1, 0.1)):
    # Define the category boundaries
    boundaries = torch.arange(0, boundary_ranges[0], boundary_ranges[1])

    # Assign categories based on the boundaries
    categories = torch.zeros_like(t, dtype=torch.long, device = t.device)
    for i in range(len(boundaries) - 1):
        mask = (t >= boundaries[i]) & (t < boundaries[i + 1])
        categories[mask] = i
    return categories

class VPSDE(nn.Module):
    r"""Creates a noise scheduler for the variance preserving (VP) SDE.

    .. math::
        \mu(t) & = \alpha(t)^2 \\
        \sigma(t)^2 & = 1 - \alpha(t)^2 + \eta^2

    Arguments:
        eps: A noise estimator :math:`\epsilon_\phi(x, t)`.
        shape: The event shape.
        alpha: The choice of :math:`\alpha(t)`.
        eta: A numerical stability term.
    """

    def __init__(
        self,
        eps: nn.Module,
        shape: Size = (),
        alpha: str = "cos",
        eta: float = 1e-3,
        model_type: str = "noise",
    ):
        super().__init__()

        self.net = eps
        self.shape = shape
        self.dims = tuple(range(-len(shape), 0))
        self.eta = eta
        self.model_type = model_type

        # alpha(t) = exp(- 1/2 \int^t beta(s) ds)
        if alpha == "lin":
            self.alphastr = "lin"
            self.alpha = lambda t: 1 - (1 - eta) * t
        elif alpha == "cos":
            self.alphastr = "cos"
            self.alpha = lambda t: torch.cos(math.acos(math.sqrt(eta)) * t) ** 2
            # self.alphajax = lambda t: jnp.cos(jnp.arccos(jnp.sqrt(eta)) * t) ** 2
        elif alpha == "exp":
            self.alphastr = "exp"
            self.alpha = lambda t: torch.exp(math.log(eta) * t**2)
        else:
            raise ValueError()

        self.net.alpha = self.alpha
        self.net.eta = self.eta
        
        self.register_buffer("device", torch.empty(()))


    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        return 0.5 * torch.log(self.alpha(t))
        # raise NotImplementedError()
        # if self.schedule == 'discrete':
        #   return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        # elif self.schedule == 'linear':
        #     return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.alphastr == 'cos':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            return 1/math.acos(math.sqrt(self.eta))*torch.acos(torch.exp(log_alpha))
        else:
            raise ValueError("Unsupported alpha {}, needs to be 'cos'".format(self.alphastr))

        # if self.schedule == 'linear':
        #     tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
        #     Delta = self.beta_0**2 + tmp
        #     return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        # elif self.schedule == 'discrete':
        #     log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
        #     t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
        #     return t.reshape((-1,))

    def mu(self, t: Tensor) -> Tensor:
        """E[x_t|x_0] = mu(t) * x_0"""
        return self.alpha(t).sqrt()

    def sigma(self, t: Tensor) -> Tensor:
        """Std[x_t|x_0] = \sigma(t)"""
        return (1 - self.mu(t) ** 2 + self.eta**2).sqrt()

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.net(x, t)


    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model.
        """
        noise = self.noise_prediction_fn(x, t)
        x0 = (x - self.sigma(t)*noise)/self.mu(t)
        return x0
    
    def base_sample(self, x):
        return torch.randn_like(x).to(self.device)
    
    def forward(self, x: Tensor, t: Tensor, train: bool = False) -> Tensor:
        r"""Samples from the perturbation kernel :math:`p(x(t) | x)`."""

        t = t.reshape(t.shape + (1,) * len(self.shape))

        eps = self.base_sample(x)
        xt = self.mu(t) * x + self.sigma(t) * eps

        if train:
            return xt, eps

        else:
            return xt

    def sample(
        self,
        steps: int = 1000,
    ) -> Tensor:
        r"""Samples from :math:`p(x(0))`.

        Arguments:
            shape: The batch shape.
            steps: The number of discrete time steps.
        """

        x = torch.randn(self.shape).to(self.device).cuda()
        x = x.reshape(-1, *self.shape)


        time = torch.linspace(1, 0, steps + 1).to(self.device).cuda()
        dt = 1 / steps

        with torch.no_grad():
            for t in tqdm(time[:-1]):
                # Predictor
                r = self.mu(t - dt) / self.mu(t)
                x = r * x + (self.sigma(t - dt) - r * self.sigma(t)) * self.net(x, t)
        print(x.shape)
        return x

    def loss(self, x: Tensor, log_loss_per_level: bool = False) -> Tensor:
        r"""Returns the denoising loss."""
        t = torch.rand(x.shape[0], dtype=x.dtype, device=x.device)
        x, output = self.forward(x, t, train=True)
        loss_weight = torch.ones(t.shape, device=x.device)

        mse = (self.net(x, t) - output).square().flatten(1).mean(1)
        if log_loss_per_level:
            t = get_time_category(t)
            return (loss_weight*mse), t
        else:
            return (loss_weight*mse).mean()

