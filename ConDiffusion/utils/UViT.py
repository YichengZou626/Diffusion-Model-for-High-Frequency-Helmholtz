# Copyright 2024 The swirl_dynamics Authors.
# Modifications made by the CAM Lab at ETH Zurich.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""3D U-Net denoiser models.

Intended for inputs with dimensions (batch, time, x, y, channels). The U-Net
stacks successively apply 2D downsampling/upsampling in space only. At each
resolution, an axial attention block (involving space and/or time) is applied.
"""

from collections.abc import Sequence
from typing import Literal, Sequence, Any, Callable, Union, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model.building_blocks.stacks.dstack_3d import DStack
from utils.model.building_blocks.stacks.ustack_3d import UStack
from utils.model.building_blocks.embeddings.fourier_emb import FourierEmbedding
from utils.model.building_blocks.layers.convolutions import ConvLayer

from utils.model.building_blocks.model_utils import default_init

Tensor = torch.Tensor


def _maybe_broadcast_to_list(
    source: Union[bool, Sequence[bool]],
    reference: Sequence[Any]
) -> List[bool]:
    """Broadcasts to a list with the same length if applicable."""
    if isinstance(source, bool):
        return [source] * len(reference)
    else:
        if len(source) != len(reference):
            raise ValueError(f"{source} must have the same length as {reference}!")
        return list(source)
    
class UViT3D(nn.Module):
    """
    UVit3D: U-Net-style 3D spatiotemporal backbone with *time-only* conditioning.

    Differences from the original UNet3D:
      - No sigma/noise-level anywhere.
      - forward() takes exactly two arguments: x (input; may already include
        channel-wise conditioning concatenated by the caller) and t (time scalar per sample).
      - Embedding is time-only via FourierEmbedding.

    Expected input shape:
      x: (batch, X, Y, Z, C)
      t: (batch,)   # 1D vector of time scalars (or steps) to embed
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_resolution: Sequence[int],
        num_channels: Sequence[int] = (128, 256, 256),
        downsample_ratio: Sequence[int] = (2, 2, 2),
        num_blocks: int = 4,
        time_embed_dim: int = 128,
        input_proj_channels: int = 128,
        output_proj_channels: int = 128,
        padding_method: str = "circular",
        dropout_rate: float = 0.0,
        use_spatial_attention: Union[bool, Sequence[bool]] = (False, False, False),
        use_position_encoding: bool = True,
        num_heads: int = 8,
        normalize_qk: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        spatial: int = 2,
    ):
        super().__init__()

        # ---------------- core hyperparams ----------------
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.spatial_resolution = spatial_resolution
        self.spatial = len(self.spatial_resolution)  # <-- add this line
        self.kernel_dim = len(spatial_resolution)  # 3 for (X,Y,Z)
        self.downsample_ratio = downsample_ratio
        self.num_blocks = num_blocks
        self.time_embed_dim = time_embed_dim
        self.input_proj_channels = input_proj_channels
        self.output_proj_channels = output_proj_channels
        self.padding_method = padding_method
        self.dropout_rate = dropout_rate
        self.use_spatial_attention = use_spatial_attention
        self.use_position_encoding = use_position_encoding
        self.num_heads = num_heads
        self.normalize_qk = normalize_qk
        self.device = device
        self.dtype = dtype

        self.use_spatial_attention = _maybe_broadcast_to_list(
            source=self.use_spatial_attention, reference=self.num_channels
        )

        # ---------------- embeddings: time only ----------------
        self.time_embedding = FourierEmbedding(
            dims=self.time_embed_dim, dtype=self.dtype, device=self.device
        )
        self.emb_channels = self.time_embed_dim

        # ---------------- UNet stacks ----------------
        self.DStack = DStack(
            in_channels=self.in_channels,
            spatial_resolution=self.spatial_resolution,
            emb_channels=self.emb_channels,
            num_channels=self.num_channels,
            num_res_blocks=len(self.num_channels) * (self.num_blocks,),
            downsample_ratio=self.downsample_ratio,
            use_spatial_attention=self.use_spatial_attention,
            num_input_proj_channels=self.input_proj_channels,
            padding_method=self.padding_method,
            dropout_rate=self.dropout_rate,
            num_heads=self.num_heads,
            use_position_encoding=self.use_position_encoding,
            normalize_qk=self.normalize_qk,
            dtype=self.dtype,
            device=self.device,
        )

        self.UStack = UStack(
            spatial_resolution=self.spatial_resolution,
            emb_channels=self.emb_channels,
            num_channels=self.num_channels[::-1],
            num_res_blocks=len(self.num_channels) * (self.num_blocks,),
            upsample_ratio=self.downsample_ratio[::-1],
            use_spatial_attention=self.use_spatial_attention[::-1],
            num_input_proj_channels=self.input_proj_channels,
            num_output_proj_channels=self.output_proj_channels,
            padding_method=self.padding_method,
            dropout_rate=self.dropout_rate,
            num_heads=self.num_heads,
            normalize_qk=self.normalize_qk,
            use_position_encoding=self.use_position_encoding,
            dtype=self.dtype,
            device=self.device,
        )

        self.norm = nn.GroupNorm(
            num_groups=min(max(self.output_proj_channels // 4, 1), 32),
            num_channels=self.output_proj_channels,
            device=self.device,
            dtype=self.dtype,
        )

        self.conv_layer = ConvLayer(
            in_channels=self.output_proj_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_dim * (3,),
            padding_mode=self.padding_method,
            padding=1,
            case=self.kernel_dim,
            kernel_init=default_init(),
            dtype=self.dtype,
            device=self.device,
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
          x: 5D tensor (B, X, Y, Z, C). If you have extra conditioning (e.g., y),
             concatenate it to x **before** calling this method along the channel dim.
          t: 1D tensor (B,) with time/step values to embed.

        Returns:
          Tensor of same spatial/feature shape as x: (B, X, Y, Z, out_channels)
        """
        # --------- sanity checks ---------
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if not isinstance(t, torch.Tensor):
            raise TypeError("t must be a torch.Tensor")

        if x.ndim != 5:
            raise ValueError(f"x must be 5D (B, X, Y, Z, C); got {x.shape}")

        if t.ndim < 1:
            t = t.expand(x.size(0))
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(
                f"t must be 1D with same batch size as x; got t.shape={t.shape}, "
                f"x.shape[0]={x.shape[0]}"
            )

        if len(self.num_channels) != len(self.downsample_ratio):
            raise ValueError(
                f"`num_channels` {self.num_channels} and `downsample_ratio` "
                f"{self.downsample_ratio} must have the same lengths!"
            )

        # --------- time embedding only ---------
        emb_time = self.time_embedding(t)  # (B, emb_channels)

        # --------- UNet pass ---------
        skips = self.DStack(x, emb_time)            # down path
        h = self.UStack(skips[-1], emb_time, skips) # up path

        h = F.silu(self.norm(h))
        h = self.conv_layer(h)
        return h

