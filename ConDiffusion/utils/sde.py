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
import numpy as np

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

        self.alphastr = "cos"
        self.alpha = lambda t: torch.cos(math.acos(math.sqrt(eta)) * t) ** 2

        self.net.alpha = self.alpha
        self.net.eta = self.eta
        
        self.register_buffer("device", torch.empty(()))

    def mu(self, t: Tensor) -> Tensor:
        """E[x_t|x_0] = mu(t) * x_0"""
        return self.alpha(t).sqrt()

    def sigma(self, t: Tensor) -> Tensor:
        """Std[x_t|x_0] = \sigma(t)"""
        return (1 - self.mu(t) ** 2 + self.eta**2).sqrt()
        
    def linear_beta_schedule(self, num_diffusion_timesteps: int) -> Tensor:
        # Linear schedule from Ho et al, scaled for general timesteps
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.02
        beta_end = scale * 0.0001
        return torch.linspace(beta_start, beta_end, num_diffusion_timesteps)

    def cosine_beta_schedule(self, t: Tensor, s=0.008) -> Tensor:
        return torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2

    def base_sample(self, x):
        return torch.randn_like(x).to(self.device)

    def forward(self, x: Tensor, t: Tensor, train: bool = False) -> Tensor:
        r"""Samples from the perturbation kernel :math:`p(x(t) | x, condition)`."""

        t = t.reshape(t.shape + (1,) * len(self.shape))  # shape match for broadcasting

        # === Add Gaussian noise ===
        eps = self.base_sample(x)
        xt = self.mu(t) * x + self.sigma(t) * eps  # diffusion with condition

        if train:
            return xt, eps
        else:
            return xt


    def sample(
        self,
        condition: torch.Tensor,  # shape: (B, C, 256)
        steps: int = 1000,
    ) -> torch.Tensor:
        r"""Samples from :math:`p(x(0))` for a batch of conditions."""
        
        batch_size = condition.shape[0]
        #print(condition.shape)
        shape = (batch_size, 1, 64, 64, 64)  # Assuming (B, 1, 256)
        x = torch.randn(shape).to(self.device).cuda()
        #print("x shape:", x.shape)
        #print("condition shape:", condition.shape)
        time = torch.linspace(1, 0, steps + 1).to(self.device).cuda()
        dt = 1 / steps

        with torch.no_grad():
            for t in tqdm(time[:-1]):
                r = self.mu(t - dt) / self.mu(t)
                x = r * x + (self.sigma(t - dt) - r * self.sigma(t)) * self.net(x, condition, t)
        return x
        

    def sample_ddim(self, condition: Tensor, steps: int = 1000) -> Tensor:
        r"""Deterministic sampling using DDIM (Denoising Diffusion Implicit Models)

        Arguments:
            condition: The input information (e.g., sound map and source field)
            steps: Number of sampling steps (e.g., 50)
        Returns:
            Sampled image x0
        """
        # Time steps
        device = self.device
        batch_size = condition.shape[0]
        #print(condition.shape)
        shape = (batch_size, 1, 64, 64, 64)  # Assuming (B, 1, 256)
        
        x = torch.randn(shape).to(self.device).cuda()
        
        # Define noise schedule (can be cosine, linear, etc.)
        t_steps = torch.linspace(0, 1, steps + 1).to(self.device).cuda()
        
        # Compute alpha, beta, alpha_bar
        beta_t_steps = self.cosine_beta_schedule(t_steps)
        alpha_t_steps = 1 - beta_t_steps
        alpha_bar_t_steps = torch.cumprod(alpha_t_steps, dim=0)

        # Define noise schedule (can be cosine, linear, etc.)
        t_steps = torch.linspace(1, 0, steps + 1).to(self.device).cuda()

        with torch.no_grad():
            for i in range(steps):
                t = t_steps[i]
                t_next = t_steps[i + 1]

                # Predict noise using the network
                eps_theta = self.net(x, condition, t)

                # Get alpha and sigma at t and t_next
                alpha_t = alpha_t_steps[i] #self.mu(t) ** 2
                alpha_next = alpha_t_steps[i+1] #self.mu(t_next) ** 2
                                
                x0 = (x - eps_theta * torch.sqrt(1 - alpha_t)) / torch.sqrt(alpha_t)
                
                c1 = 1 * torch.sqrt((1 - alpha_t / alpha_next) * (1 - alpha_next) / (1 - alpha_t))
                c2 = torch.sqrt((1 - alpha_next) - c1 ** 2)

                x = torch.sqrt(alpha_next) * x0 + c1 * torch.randn_like(x) + c2 * eps_theta
                
                # === Logging values ===
                print(f"Step {i}")
                print(f"  alpha_t      = {alpha_t.item():.8f}")
                print(f"  alpha_next   = {alpha_next.item():.8f}")
                print(f"  c1 (eta term)= {c1.item():.8f}")
                print(f"  c2           = {c2.item():.8f}")
                print(f"  Max abs(x0)       = {torch.max(torch.abs(x0)).item():.6f}")
                print(f"  Max abs(x)        = {torch.max(torch.abs(x)).item():.6f}")
                print(f"  Max abs(eps_theta)= {torch.max(torch.abs(eps_theta)).item():.6f}")


        return x



    def sample_ddpm(self, condition: Tensor, steps: int = 1000, schedule_name: str = "linear") -> Tensor:
        """
        model: the trained noise prediction model ε_θ(x_t, t)
        steps: number of timesteps (T)
        shape: shape of final output (e.g. image)
        """

        device = self.device
        batch_size = condition.shape[0]
        #print(condition.shape)
        shape = (batch_size, 1, 64, 64, 64)  # Assuming (B, 1, 256)

        x = torch.randn(shape).to(self.device).cuda()
        
        max_x = torch.max(torch.abs(x)).item()
        print(f"  Max abs(x): {max_x:.4f}")

        # Define noise schedule (can be cosine, linear, etc.)
        t_steps = torch.linspace(1, 0, steps + 1).to(self.device).cuda()

        # Choose noise schedule
        if schedule_name == "linear":
            beta_t_steps = self.linear_beta_schedule(steps + 1)
            #beta_t_steps = torch.where(beta_t_steps < 1.0, beta_t_steps, torch.ones_like(beta_t_steps))
            alpha_t_steps = 1.0 - beta_t_steps
        elif schedule_name == "cosine":
            beta_t_steps = self.cosine_beta_schedule(t_steps)
            alpha_t_steps = 1.0 - beta_t_steps
        else:
            raise ValueError(f"Unknown noise schedule: {schedule_name}")

        alpha_bar_t_steps = torch.cumprod(alpha_t_steps, dim=0)
        
        
        with torch.no_grad():
            for i in range(1, steps):
                t = t_steps[i]

                # Compute alpha, beta, alpha_bar
                alpha_t = torch.abs(alpha_t_steps[i])
                beta_t = beta_t_steps[i]
                alpha_bar_t = torch.abs(alpha_bar_t_steps[i])

                # Print schedule values
                print(f"Step {i+1}/{steps}")
                print(f"  t = {t.item():.8f}")
                print(f"  alpha_t = {alpha_t.item():.8f}")
                print(f"  beta_t = {beta_t.item():.8f}")
                print(f"  alpha_bar_t = {alpha_bar_t.item():.8f}")

                # Predict noise
                eps_theta = self.net(x, condition, t)
                print(f"  eps_theta shape: {eps_theta.shape}")
                max_eps = torch.max(torch.abs(eps_theta)).item()
                print(f"  Max abs(eps_theta): {max_eps:.4f}")

                # Compute posterior mean
                coef1 = 1 / torch.sqrt(torch.tensor(alpha_t))
                coef2 = beta_t / torch.sqrt(torch.tensor(1 - alpha_bar_t))
                mean = coef1 * (x - coef2 * eps_theta)
                max_mean = torch.max(torch.abs(mean)).item()
                print(f"  coef1 = {coef1.item():.8f}")
                print(f"  coef2 = {coef2.item():.8f}")
                print(f"  Max abs(mean): {max_mean:.4f}")

                # Inject noise if not final step
                if t > 0:
                    z = torch.randn_like(x)
                    max_z = torch.max(torch.abs(z)).item()
                    print(f"  Max abs(z): {max_z:.4f}")
                    sigma_t = torch.sqrt(beta_t)
                    print(f"  sigma_t = {sigma_t.item():.8f}")

                    x = mean + sigma_t * z / 10
                else:
                    x = mean

                # Monitor updated x
                max_abs = torch.max(torch.abs(x)).item()
                print(f"  Max abs(x after update): {max_abs:.4f}")
                print("-" * 50)  # separator for readability


        return x

    
    
    def loss(self, x: Tensor, condition: Tensor, log_loss_per_level: bool = False) -> Tensor:
        r"""Returns the denoising loss."""
        t = torch.rand(x.shape[0], dtype=x.dtype, device=x.device)
        x, output = self.forward(x, t, train=True)
        loss_weight = torch.ones(t.shape, device=x.device)

        mse = (self.net(x, condition, t) - output).square().flatten(1).mean(1)
        if log_loss_per_level:
            t = get_time_category(t)
            return (loss_weight*mse), t
        else:
            return (loss_weight*mse).mean()
