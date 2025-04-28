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


class ConditionedScoreUNet(nn.Module):
    r"""Creates a U-Net score network with a forcing channel."""

    def __init__(
        self,
        net: nn.Module,
        in_features: int = 64,
        out_features: int = 64,
        add_noise: bool = False,
        size: int = 64,
        use_forcing: bool = True,
        **kwargs):
        super().__init__()

        self.use_forcing = use_forcing
        forcing_channels = 1 if self.use_forcing else 0

        # Setup main network
        self.network = net(in_features + out_features + forcing_channels, out_features + forcing_channels)
        self.condition = None
        self.condition_dim = in_features
        self.mask_dim = self.condition_dim // 2
        self.in_features = in_features + out_features + forcing_channels
        self.out_features = out_features + forcing_channels
        self.alpha = None
        self.add_noise = add_noise
        self.eta = None

        # Create forcing field if needed
        if self.use_forcing:
            domain = 2 * torch.pi / size * (torch.arange(size) + 0.5)
            forcing = torch.sin(4 * domain).expand(1, size, size).clone()
            self.register_buffer("forcing", forcing)

    def set_condition(self, condition: Tensor):
        dims = self.network.spatial + 1
        assert condition.shape[-dims] == self.condition_dim
        self.condition = condition

    def set_zero_condition(self):
        dim = self.network.spatial
        self.condition = torch.zeros(1, self.condition_dim, *([1] * dim)).to(next(self.network.parameters()).device)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        dims = self.network.spatial + 1
        y = x.reshape(-1, *x.shape[-dims:])

        assert self.condition is not None, "Condition not set. Please call set_condition() first."

        # Expand condition to match input
        condition = self.condition.expand(y.shape[0], -1, *y.shape[-self.network.spatial:])

        current_condition = condition

        # Concatenate condition and input
        y = torch.cat([y, current_condition], dim=1)

        # Add forcing if enabled
        if self.use_forcing:
            x, f = broadcast(x, self.forcing, ignore=3)
            forcing = f.expand(x.shape[0], -1, *x.shape[-2:])  # Expand to batch size
            y = torch.cat((y, forcing), dim=-3)

        # Pass through network
        output = self.network(y, t)
        output = output.reshape(list(x.shape)[:-self.network.spatial-1] + [self.out_features] + list(x.shape[-self.network.spatial:]))

        # Remove forcing channel from output if used
        if self.use_forcing:
            output = output[..., :-1, :, :]

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
        spatial,
        condition_dim=0,
):
    # Partially create neural network
    net = instantiate(net, _partial_=True)

    # Create score wrapper to combine with time context
    score = instantiate(
        score,
        net=net,
        features=spatial * window,
        in_features=spatial*condition_dim,
        out_features=spatial*window,
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

    if not cfg.amortized:
        condition_dim = 0
    else:
        condition_dim = cfg.window*2

    score = make_score(
        score=cfg.score,
        net=cfg.net,
        window=cfg.window,
        spatial=cfg.data.spatial,
        condition_dim=condition_dim,
    )
    score.load_state_dict(state)
    return score
