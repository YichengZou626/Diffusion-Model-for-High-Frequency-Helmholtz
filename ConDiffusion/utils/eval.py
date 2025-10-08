"""Evaluation metrics"""

import torch
import numpy as np
from torch import Tensor
from pathlib import Path

def get_training_dir(current_dir, cfg):
    """returns path to associated training directory"""
    path = Path(current_dir)
    parent_dir = path.parent
    if parent_dir.name != cfg.name:  # if override_dirname is empty
        lines = parent_dir.name.split(",")
        new_lines = []
        for line in lines:
            # has_opening_bracket = '[' in line
            # has_closing_bracket = ']' in line
            splits = line.split('=')
            if len(splits) == 1: #NOTE: means it's due a list with comma
                new_lines[-1] += ',' + line
            else:
                if 'eval' not in splits[0]:
                    new_lines.append(line)
        training_dir = ','.join(new_lines)
        training_path = parent_dir.parent.joinpath(training_dir)
    else:
        training_path = parent_dir
    training_path = training_path.joinpath(str(cfg.seed))
    return training_path


def get_sampler(cfg):
    # Sampler
    sampler = DPM(
        steps=cfg.eval.sampling.steps,
        corrections=cfg.eval.sampling.corrections,
        tau=cfg.eval.sampling.tau,
        skip_type=cfg.sampler.skip_type,
        order=cfg.sampler.order,
        correcting_x0_fn=cfg.sampler.correcting_x0_fn,
        algorithm_type=cfg.sampler.alg_type,
        denoise_to_zero=cfg.sampler.denoise_to_zero)
    return sampler
