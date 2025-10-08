import h5py
import json
import math
import numpy as np
import ot
import random
import torch
import os
import gdown
from pathlib import Path
from torch import Tensor
from tqdm import trange
from typing import *
from sklearn.preprocessing import MinMaxScaler

def load_data(file: Path) -> Tensor:
    if not os.path.exists(file):
        os.makedirs('/'.join(file.split('/')[:-1]), exist_ok=True)
        dataset_name, split = file.split('/')[-2:]
        url = DATASET_TO_URL[dataset_name][split]
        gdown.download(url, file, quiet=False)

    with h5py.File(file, mode="r") as f:
        input = f["x"][()]
        target = f["y"][()]
        data = np.concatenate([input, target], axis=1)

    # After: data = np.concatenate([input, target], axis=1)  # [N,4,256,256]
    #N, C, H, W = data.shape
    #assert (H, W) == (256, 256), f"Expected 256x256, got {H}x{W}"
    #data = np.ascontiguousarray(data).reshape(N, C, 64, 32, 32)  # -> [N,4,64,32,32]

    return {'data': torch.from_numpy(data).float()}

