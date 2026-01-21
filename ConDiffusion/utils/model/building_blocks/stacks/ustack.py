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

"""Upsampling Stack for 2D Data Dimensions"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import ToPILImage, ToTensor

from typing import Tuple, Literal, Sequence, Any, Callable, Union, Optional
import torch
import torch.nn as nn
from utils.model.building_blocks.layers.upsample import (
    ChannelToSpace,
    LearnablePixelShuffle3D,
    TransposeConv3D,
)
from utils.model.building_blocks.layers.residual import CombineResidualWithSkip
from utils.model.building_blocks.layers.convolutions import ConvLayer
from utils.model.building_blocks.blocks.convolution_blocks import ConvBlock, ResConv1x
from utils.model.building_blocks.blocks.attention_block import AttentionBlock
from utils.model.building_blocks.model_utils import reshape_jax_torch, default_init

Tensor = torch.Tensor


class UStack(nn.Module):
    """Upsampling Stack.

    Takes in features at intermediate resolutions from the downsampling stack
    as well as final output, and applies upsampling with convolutional blocks
    and combines together with skip connections in typical UNet style.
    Optionally can use self attention at low spatial resolutions.

    Attributes:
        num_channels: Number of channels at each resolution level.
        num_res_blocks: Number of resnest blocks at each resolution level.
        upsample_ratio: The upsampling ration between levels.
        padding: Type of padding for the convolutional layers.
        dropout_rate: Rate for the dropout inside the transformed blocks.
        use_attention: Whether to use attention at the coarser (deepest) level.
        num_heads: Number of attentions heads inside the attention block.
        channels_per_head: Number of channels per head.
        dtype: Data type.
    """

    def __init__(
        self,
        spatial_resolution: Sequence[int],
        emb_channels: int,
        num_channels: Sequence[int],
        num_res_blocks: Sequence[int],
        upsample_ratio: Sequence[int],
        padding_method: str = "circular",
        dropout_rate: float = 0.0,
        use_attention: bool = False,
        num_input_proj_channels: int = 128,
        num_output_proj_channels: int = 128,
        num_heads: int = 8,
        channels_per_head: int = -1,
        normalize_qk: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        super(UStack, self).__init__()

        self.kernel_dim = len(spatial_resolution)
        self.emb_channels = emb_channels
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.upsample_ratio = upsample_ratio
        self.padding_method = padding_method
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.num_input_proj_channels = num_input_proj_channels
        self.num_output_proj_channels = num_output_proj_channels
        self.num_heads = num_heads
        self.channels_per_head = channels_per_head
        self.normalize_qk = normalize_qk
        self.dtype = dtype
        self.device = device

        # Calculate channels for the residual block
        in_channels = []

        # calculate list of upsample resolutions
        list_upsample_resolutions = [spatial_resolution]
        for level, channel in enumerate(self.num_channels):
            downsampled_resolution = tuple(
                [
                    int(res / self.upsample_ratio[level])
                    for res in list_upsample_resolutions[-1]
                ]
            )
            list_upsample_resolutions.append(downsampled_resolution)
        list_upsample_resolutions = list_upsample_resolutions[::-1]
        list_upsample_resolutions.pop()

        self.residual_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.res_conv_blocks = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        for level, channel in enumerate(self.num_channels):
            self.conv_blocks.append(nn.ModuleList())
            self.residual_blocks.append(nn.ModuleList())

            for block_id in range(self.num_res_blocks[level]):
                if block_id == 0 and level > 0:
                    in_channels.append(self.num_channels[level - 1])
                else:
                    in_channels.append(channel)

                self.residual_blocks[level].append(
                    CombineResidualWithSkip(
                        residual_channels=in_channels[-1],
                        skip_channels=channel,
                        kernel_dim=self.kernel_dim,
                        project_skip=in_channels[-1] != channel,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
                self.conv_blocks[level].append(
                    ConvBlock(
                        in_channels=in_channels[-1],
                        out_channels=channel,
                        emb_channels=self.emb_channels,
                        kernel_size=self.kernel_dim * (3,),
                        padding_mode=self.padding_method,
                        padding=1,
                        case=self.kernel_dim,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )

                if self.use_attention and level == 0:
                    self.attention_blocks.append(
                        AttentionBlock(
                            in_channels=channel,
                            num_heads=self.num_heads,
                            normalize_qk=self.normalize_qk,
                            dtype=self.dtype,
                            device=self.device,
                        )
                    )
                    self.res_conv_blocks.append(
                        ResConv1x(
                            in_channels=channel,
                            hidden_layer_size=channel * 2,
                            out_channels=channel,
                            kernel_dim=1,  # 1D due to token shape (bs, l, c)
                            dtype=self.dtype,
                            device=self.device,
                        )
                    )

            # Upsampling step
            up_ratio = self.upsample_ratio[level]
            self.conv_layers.append(
                ConvLayer(
                    in_channels=channel,
                    out_channels=up_ratio**self.kernel_dim * channel,
                    kernel_size=self.kernel_dim * (3,),
                    padding_mode=self.padding_method,
                    padding=1,
                    case=self.kernel_dim,
                    kernel_init=default_init(1.0),
                    dtype=self.dtype,
                    device=self.device,
                )
            )

            # For higher or lower input dimensions than 2 use ChannelToSpace from upsample.py
            self.upsample_layers.append(
                # upscaling spatial dimensions by rearranging channel data
                nn.PixelShuffle(up_ratio)  # Only for 2D data input
            )

        # DStack Input - UStack Output Residual Connection
        self.res_skip_layer = CombineResidualWithSkip(
            residual_channels=self.num_channels[-1],
            skip_channels=self.num_input_proj_channels,
            project_skip=(self.num_channels[-1] != self.num_input_proj_channels),
            dtype=self.dtype,
            device=self.device,
        )

        # Add output layer
        self.conv_layers.append(
            ConvLayer(
                in_channels=self.num_channels[-1],
                out_channels=self.num_output_proj_channels,
                kernel_size=self.kernel_dim * (3,),
                padding_mode=self.padding_method,
                padding=1,
                case=self.kernel_dim,
                kernel_init=default_init(1.0),
                dtype=self.dtype,
                device=self.device,
            )
        )

    def forward(self, x: Tensor, emb: Tensor, skips: list[Tensor]) -> Tensor:
        assert (
            len(self.num_channels)
            == len(self.num_res_blocks)
            == len(self.upsample_ratio)
        )
        h = x

        for level, channel in enumerate(self.num_channels):
            for block_id in range(self.num_res_blocks[level]):
                # Residual
                h = self.residual_blocks[level][block_id](residual=h, skip=skips.pop())
                # Convolution Blocks
                h = self.conv_blocks[level][block_id](h, emb)
                # Spatial Attention Blocks
                if self.use_attention and level == 0:
                    h = reshape_jax_torch(
                        h, kernel_dim=self.kernel_dim
                    )  # (bs, width, height, c)
                    b, *hw, c = h.shape

                    h = self.attention_blocks[block_id](h.reshape(b, -1, c))
                    h = reshape_jax_torch(
                        self.res_conv_blocks[block_id](
                            reshape_jax_torch(h, kernel_dim=1)
                        ),
                        kernel_dim=1,
                    ).reshape(b, *hw, c)
                    h = reshape_jax_torch(
                        h, kernel_dim=self.kernel_dim
                    )  # (bs, c, width, height)

            # Upsampling Block
            h = self.conv_layers[level](h)
            # Shift channels to increase the resolution (only valid for 2D input data)
            h = self.upsample_layers[level](h)

        # Output - Input Residual Connection
        h = self.res_skip_layer(residual=h, skip=skips.pop())
        # Output Layer
        h = self.conv_layers[-1](h)

        return h

