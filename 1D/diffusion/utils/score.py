"""Score modules"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import *
from zuko.utils import broadcast
from hydra.utils import instantiate
from pathlib import Path
from omegaconf import OmegaConf
import pdb
import torch.nn.functional as F
import math
import torch
import torch.nn.functional as F

# === Define 1D Gaussian Kernel ===
def gaussian_kernel1d(kernel_size=5, sigma=1.0):
    x = torch.arange(kernel_size) - kernel_size // 2
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # Normalize
    return kernel.view(1, 1, -1)  # shape: (1, 1, kernel_size)


def get_positional_embedding(size, embedding_dim, device, max_positions=10000):
    """
    Generate 1D sinusoidal position embedding for spatial positions [0, ..., size-1]
    Returns: Tensor of shape [embedding_dim, size]
    """
    positions = torch.arange(size, dtype=torch.float32, device=device)  # [size]
    half_dim = embedding_dim // 2
    scale = math.log(max_positions) / (half_dim - 1)
    freqs = torch.exp(-scale * torch.arange(half_dim, dtype=torch.float32, device=device))  # [half_dim]
    angles = positions[:, None] * freqs[None, :]  # [size, half_dim]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [size, embed_dim]
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')  # pad to even size
    return emb.T  # shape: [embedding_dim, size]



class ConditionedScoreUNet(nn.Module):
    r"""Creates a U-Net score network with a forcing channel."""

    def __init__(
        self,
        net: nn.Module,
        in_features: int = 64,
        out_features: int = 64,
        add_noise: bool = False,
        size: int = 64,
        use_forcing: bool = False,
        **kwargs):
        super().__init__()

        self.use_forcing = use_forcing
        forcing_channels = 1 if self.use_forcing else 0

        # Setup main network
        self.network = net(in_features+1, out_features)
        self.condition = None
        self.condition_dim = in_features
        self.mask_dim = self.condition_dim // 2
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = None
        self.add_noise = add_noise
        self.eta = None

    def forward(self, x: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        dims = self.network.spatial + 1  # dims = 2 for 1D input
        y = x.reshape(-1, *x.shape[-dims:])  # shape: [B, 1, 512]
        size = y.shape[-1]  # 512

        # --- Concatenate condition (assumed to be already [B, C, 512])
        y = torch.cat([y, condition], dim=1)

        # --- Pass through the network
        output = self.network(y, t)

        # --- Reshape back to original format
        output_shape = list(x.shape)[:-self.network.spatial - 1] + [self.out_features] + list(x.shape[-self.network.spatial:])
        output = output.reshape(output_shape)

        return output


class ConditionedScoreHNO(nn.Module):
    r"""Creates a U-Net score network with a forcing channel."""

    def __init__(
            self,
            net: nn.Module,
            in_features: int = 64,
            out_features: int = 64,
            add_noise: bool = False,
            size: int = 64,
            use_forcing: bool = False,
            **kwargs):
        super().__init__()

        self.use_forcing = use_forcing
        forcing_channels = 1 if self.use_forcing else 0

        # Setup main network
        self.network = net(in_features, out_features)
        self.condition = None
        self.condition_dim = in_features
        self.mask_dim = self.condition_dim // 2
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = None
        self.add_noise = add_noise
        self.eta = None

    def forward(self, x: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        dims = self.network.spatial + 1
        y = x.reshape(-1, *x.shape[-dims:])

        y = torch.cat([y, condition], dim=1)

        # Pass through network
        output = self.network(y, t)


        return output
        
        
class ScoreWrapper(nn.Module):
    """Just a wrapper for our score network that is passed in the constructor.
        The forward just call the forward of the score network
    """

    def __init__(self, score: nn.Module):
        super().__init__()

        self.score = score

    def forward(
        self,
        x: Tensor,  # (B, L, C, H, W)
        t: Tensor,  # ()
    ) -> Tensor:
        return self.score(x.transpose(1, 2), t).transpose(1, 2)


class ScoreNet(nn.Module):
    r"""Creates a score network for a Markov chain.

    Arguments:
        features: The number of features.
        order: The order of the Markov chain.
    """

    def __init__(self, kernel, order: int):
        super().__init__()

        self.order = order
        self.kernel = kernel
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        dims = self.kernel.network.spatial + 2
        num_channels = x.shape[-(dims - 1)]

        x = self.unfold(x, self.order)

        assert (
            x.shape[-(dims - 1)] == (2 * self.order + 1) * num_channels
        ), f"Dimensions {x.shape} are not consistent with the window size {2*self.order+1}"

        s = self.kernel(x, t)

        assert (
            s.shape[-(dims - 1)] == (2 * self.order + 1) * num_channels
        ), f"Dimensions {s.shape} are not consistent with the window size {2*self.order+1}"

        s = self.fold(s, self.order)
        return s

    # the tag is just compiling the function when it is first called during tracing
    @staticmethod
    @torch.jit.script_if_tracing
    def unfold(x: Tensor, order: int) -> Tensor:
        """
        This method take the batch of trajectories, and return all the psudo markov
        blanket described by Algorithm 2 in the paper.
        So it just create the following:
        - x_{1:2k+1}(t)
        - x_{i−k:i+k}(t) for i = k + 2 to L − k − 1
        - x_{L−2k:L}(t)

        These are all the input to our score network that are used to compute the approximate score.
        """

        x = x.unfold(1, 2 * order + 1, 1)
        x = x.movedim(-1, 2)
        x = x.flatten(2, 3)

        return x

    @staticmethod
    @torch.jit.script_if_tracing
    def fold(x: Tensor, order: int) -> Tensor:
        """
        Function that given all the scores computed in each markov blanket and
        compose the approximated score as described in Algorithm 2
        """
        x = x.unflatten(2, (2 * order + 1, -1))

        return torch.cat(
            (
                x[:, 0, :order],
                x[:, :, order],
                x[:, -1, -order:],
            ),
            dim=1,
        )

def make_score(
        score,
        net,
        window,
        condition_dim=0,
):
    # Partially create neural network
    net = instantiate(net, _partial_=True)

    # Create score wrapper to combine with time context
    score = instantiate(
        score,
        net=net,
        features=window,
        in_features=condition_dim,
        out_features=1,
    )

    # Construct full score network from markov blanket scores
    return ScoreNet(
        kernel=score,
        order=window // 2,
    )


def load_score(file: Path, device: str = "cpu", **kwargs) -> nn.Module:
    state = torch.load(file, map_location=device)
    cfg = OmegaConf.load(file.parent.parent.joinpath(".hydra/config.yaml"))
    cfg.update(kwargs)

    condition_dim = cfg.window

    score = make_score(
        score=cfg.score,
        net=cfg.net,
        window=cfg.window,
        condition_dim=condition_dim,
    )
    score.load_state_dict(state)
    return score
