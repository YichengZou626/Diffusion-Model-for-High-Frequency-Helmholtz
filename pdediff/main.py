import torch
import os
import hydra
import math

import pdediff.utils as sda_utils
import pdediff.eval as pdediff_eval
import pdediff.sampling as sampling

from omegaconf import OmegaConf
from hydra.utils import call, instantiate
from pathlib import Path

from pdediff.sde import VPSDE
from pdediff.loop import loop
from pdediff.utils.loggers import LoggerCollection
from pdediff.sampler.utils import get_sampler
from pdediff.score import make_score, load_score
from pdediff.nn.ema import ExponentialMovingAverage
from pdediff.utils.data_preprocessing import get_true_x, get_conditioning, get_space_time_conditioning, get_space_conditioning
import pdediff.rollout as rollout
import pdediff.guidance as guidance
import pdediff.viz.plotting as viz
import matplotlib.pyplot as plt
import numpy as np
from pdediff.mcs import curl
from pdediff.mcs import KolmogorovFlow


def check_experiment_name(name: str, amortized: bool = False):
    if amortized:
        return (
                (
                    ("conditional" in name)
                ) 
            and 
                (
                    ("helmholtz" in name)
                )
            )
        
    return (
            ("helmholtz" in name)
        )


@hydra.main(config_path="config", config_name="main", version_base="1.3.2")
def main(cfg):
    os.environ["HYDRA_FULL_ERROR"] = "1"
    cfg_to_save = OmegaConf.to_container(cfg, resolve=True)

    current_dir = os.getcwd()
    ckpt_path = Path(os.path.join(current_dir, cfg.ckpt_dir, "score_last.pth"))
    ckpt_path_ema = Path(os.path.join(current_dir, cfg.ckpt_dir, "score_ema.pth"))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path_latest = Path(os.path.join(current_dir, cfg.ckpt_dir, "latest.pth"))

    loggers = [instantiate(logger_config) for logger_config in cfg.logger.values()]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(cfg_to_save)

    window = cfg.window

    if cfg.amortized:
        condition_dim = window
    else:
        condition_dim = 0

    # Data
    print("Loading data")
    
    if cfg.mode in ["train", "all"]:
        trainset = sda_utils.load_data(os.path.join(cfg.data.path, "train.h5"), 
                            window=window, 
                            spatial=cfg.data.spatial)
        print("Train data min max", trainset['data'].min(), trainset['data'].max(), trainset['data'].shape)
        validset = sda_utils.load_data(os.path.join(cfg.data.path, "valid.h5"), 
                            window=window, 
                            spatial=cfg.data.spatial)
        print("Valid data min max", validset['data'].min(), validset['data'].max(), validset['data'].shape)
    if cfg.mode in ["eval", "all"]:
        test_dataset = sda_utils.load_dataset(os.path.join(cfg.data.path, "test.h5"))
        print("Test data min max",test_dataset['data'].min(), test_dataset['data'].max(), test_dataset['data'].shape)

    # load_dataset

    # Network
    print("Making the score")

    score = make_score(
        cfg.score,
        cfg.net,
        window,
        cfg.data.spatial,
        condition_dim=condition_dim*2,
    )

    shape = (window * cfg.data.spatial, *cfg.data.grid_size)

    sde = VPSDE(
        eps=score.kernel,
        shape=shape,
        model_type=cfg.model_type,
    ).cuda()

    ema = ExponentialMovingAverage(score.parameters(), decay=cfg.ema_decay)
    
    optimizer = instantiate(cfg.optim, params=sde.parameters())

    scheduler = cfg.scheduler_name
    epochs = cfg.epochs
    if scheduler == "linear":
        lr = lambda t: 1 - (t / epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
    elif scheduler == "cosine":
        # lr = lambda t: (1 + math.cos(math.pi * t / epochs)) / 2
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, 
            T_max=epochs, 
            eta_min=1e-6)
    elif scheduler == "exponential":
        lr = lambda t: math.exp(-7 * (t / epochs) ** 2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
    else:
        raise ValueError()
    
    epoch_offset = 0

    if cfg.mode in ["train", "all"]:
        generator = loop(
            sde=sde,
            trainset=trainset,
            validset=validset,
            optimizer=optimizer,
            scheduler=scheduler,
            device="cuda",
            ema=ema,
            data_spatial=cfg.data.spatial,
            **cfg,
        )

        # Initialize lists to store loss values for plotting
        train_losses = []
        valid_losses = []
        iterations = []

        if cfg.log_loss_per_level:
            for i, (loss_train, loss_valid, lr, losses_train, losses_valid, epoch) in enumerate(generator):
                # Store losses for plotting
                train_losses.append(loss_train)
                valid_losses.append(loss_valid)
                iterations.append(i)

                logger.log_metrics(
                    {
                        "loss_train": loss_train,
                        "loss_valid": loss_valid,
                        "lr": lr,
                        "loss_train0": losses_train[0],
                        "loss_train1": losses_train[1],
                        "loss_train2": losses_train[2],
                        "loss_train3": losses_train[3],
                        "loss_train4": losses_train[4],
                        "loss_train5": losses_train[5],
                        "loss_train6": losses_train[6],
                        "loss_train7": losses_train[7],
                        "loss_train8": losses_train[8],
                        "loss_train9": losses_train[9],
                        "loss_train10": losses_train[10],
                        "loss_valid0": losses_valid[0],
                        "loss_valid1": losses_valid[1],
                        "loss_valid2": losses_valid[2],
                        "loss_valid3": losses_valid[3],
                        "loss_valid4": losses_valid[4],
                        "loss_valid5": losses_valid[5],
                        "loss_valid6": losses_valid[6],
                        "loss_valid7": losses_valid[7],
                        "loss_valid8": losses_valid[8],
                        "loss_valid9": losses_valid[9],
                        "loss_valid10": losses_valid[10],
                    }
                )
                if (i + 1) % 100 == 0: 
                    checkpoint = {'epoch': i + 1 + epoch_offset,
                                'model': sde.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(), 
                                'ema': ema.state_dict()}
                    torch.save(checkpoint, ckpt_path_latest)              
        else:
            for i, (loss_train, loss_valid, lr, epoch) in enumerate(generator):
                # Store losses for plotting
                train_losses.append(loss_train)
                valid_losses.append(loss_valid)
                iterations.append(i)

                logger.log_metrics(
                    {
                        "loss_train": loss_train,
                        "loss_valid": loss_valid,
                        "lr": lr,
                    }
                )
                if (i + 1) % 100 == 0: 
                    checkpoint = {'epoch': i + 1 + epoch_offset,
                                'model': sde.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(), 
                                'ema': ema.state_dict()}
                    torch.save(checkpoint, ckpt_path_latest)

        # Plot final loss curves using iterations instead of epochs
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, train_losses, label='Training Loss')
        plt.plot(iterations, valid_losses, label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.savefig('overall_loss_plot.png')
        plt.close()

        # Save
        print(ckpt_path)
        torch.save(score.state_dict(), ckpt_path)
        with ema.average_parameters(score.parameters()):
            torch.save(score.state_dict(), ckpt_path_ema)

    if cfg.mode in ["eval", "all"]:
        if cfg.mode == "eval":
            print("Loading the ckpt")
            training_dir = pdediff_eval.get_training_dir(current_dir, cfg)
            ckpt_path = Path(os.path.join(training_dir, cfg.ckpt_dir, cfg.eval.load_model_name))
            score = load_score(ckpt_path)
            print('Checkpoint path', ckpt_path)

        sampler = get_sampler(cfg)

        if "conditional" not in cfg.name:
            for run_idx in range(10):
                print(f"Run {run_idx + 1}")

                true_x = get_true_x(test_dataset, cfg)
                test_batch_size = cfg.eval.forecast.test_batch_size

                # Get conditioning information
                y_true, mask = get_conditioning(true_x, cfg)

                if cfg.eval.rollout_type=="all_at_once":
                    sampled_x = sampling.get_cond_aao_samples(score, y_true, sampler, cfg, logger, mask)
                elif cfg.eval.rollout_type=="autoregressive":
                    sampled_x = sampling.get_cond_ar_samples(score, y_true, sampler, cfg, logger, mask)
                else:
                    raise ValueError(f"Unsupported rollout_type {cfg.eval.rollout_type}, needs to be all_at_once or autoregressive")

                plot_fn = viz.plot_2d_trajectories

                test_plot = plot_fn(true_x[:], 0, False)
                sample_plot = plot_fn(sampled_x.squeeze(2)[:], run_idx + 1, False)


        
        else:

            # Loop to run the process 10 times
            for run_idx in range(10):
                print(f"Run {run_idx + 1}")

                likelihood = guidance.Gaussian
                likelihood_std = cfg.eval.guidance.std
                gamma = cfg.eval.guidance.gamma
                if cfg.eval.guidance.type == "SDA":
                    guidance_type = guidance.SDA

                rollout_sampler = rollout.AmortizedRollout(
                    score=score,
                    state_shape=tuple(cfg.data.state_shape),
                    sampler=sampler,
                    conditioned_frame=cfg.eval.forecast.conditioned_frame,
                    predictive_horizon=cfg.eval.forecast.predictive_horizon,
                    likelihood=likelihood,
                    guidance=guidance_type,
                    likelihood_std=likelihood_std,
                    gamma=gamma,
                    **cfg.eval.sampling,
                )

                print('Generating conditional samples')

                test_batch_size = cfg.eval.forecast.test_batch_size

                true_x = test_dataset['data'][:cfg.eval.forecast.n_samples, :cfg.eval.forecast.trajectory_length]

                print(f'True test samples size: {true_x.shape}')

                plot_fn = viz.plot_2d_trajectories

                test_plot_to_log = plot_fn(true_x[:], 0, True)

                all_conditional_samples = []


                def prepare_initial_condition(x):
                    batch_size = x.shape[0]
                    conditioning = x[:, :cfg.window].reshape((batch_size, -1, *cfg.data.state_shape[1:]))
                    mask = torch.zeros_like(conditioning)
                    if cfg.eval.guidance.std_init > 0:
                        conditioning = torch.normal(conditioning, cfg.eval.guidance.std_init)
                    mask[:, :cfg.eval.forecast.conditioned_frame * cfg.data.state_shape[0]] = 1.0
                    return torch.cat([mask, mask * conditioning], dim=1)


                initial_conditions = prepare_initial_condition(true_x)
                observations = None
                observations_mask = None

                initial_conditions = initial_conditions.split(test_batch_size)

                if observations is None:
                    observations = [None] * len(initial_conditions)
                    observations_mask = [None] * len(initial_conditions)
                else:
                    observations = observations.split(test_batch_size)
                    observations_mask = observations_mask.split(test_batch_size)

                for batch_initial_conditions, obs, obs_mask in zip(initial_conditions, observations, observations_mask):
                    conditional_samples = rollout_sampler.sample_traj(
                        cfg.eval.forecast.trajectory_length,
                        seed=cfg.seed,
                        batch_shape=(test_batch_size,),
                        conditions=batch_initial_conditions,
                        obs=obs,
                        obs_mask=obs_mask
                    )
                    all_conditional_samples += [conditional_samples]

                all_conditional_samples = torch.cat(all_conditional_samples, dim=0)
                plot_to_log = plot_fn(all_conditional_samples.squeeze(2)[:], run_idx+1, True)


        logger.close()


if __name__ == "__main__":
    main()
