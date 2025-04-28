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

def load_dataset(file: Path) -> Tensor:
    if not os.path.exists(file):
        os.makedirs('/'.join(file.split('/')[:-1]), exist_ok=True)
        dataset_name, split = file.split('/')[-2:]
        url = DATASET_TO_URL[dataset_name][split]
        gdown.download(url, file, quiet=False)

    with h5py.File(file, mode="r") as f:
        data = f["x"][:]

    return {'data': torch.from_numpy(data).float()}


def load_data(file: Path,
              window: int = None,
              spatial: int = 2) -> Tensor:
    """
    The window argument prepared the pseudo markov blanket
    used for approxiamting the score. However I think there is something
    strange going on here. Or I am just confused.

    NOTE: If we condier a specific window size, shouldn't we be
    symmetric. So I have the feeling that data = data.unfold(1, window, 1)
    should be data = data.unfold(1, 2*window+1, 1). But in this case it's not the case.

    NOTE (2): windows should be an odd number only, otherwise it is not running. Or at least
                this is what is happening in the lorenz experiment. So k = window // 2

    """

    data_dict = load_dataset(file)

    if window is None:
        pass
    else:
        data = data_dict['data'].unfold(1, window, 1)
        data = data.movedim(-1, 2)
        data = data.flatten(2, 3)
        data = data.flatten(0, 1)
        data_dict['data'] = data

    return data_dict
