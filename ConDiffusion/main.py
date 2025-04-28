import torch
import os
os.environ["WANDB_MODE"] = "offline"
import hydra
import math
from utils.data_processing import load_data, load_dataset
from utils.score import make_score, load_score
from utils.sde import VPSDE
from utils.ema import ExponentialMovingAverage
from utils.trainer import loop
from utils.eval import get_training_dir, get_sampler
from omegaconf import OmegaConf
from hydra.utils import call, instantiate
from pathlib import Path
from utils.loggers import LoggerCollection
import matplotlib.pyplot as plt
import numpy as np


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

    condition_dim = 3

    # Data
    print("Loading data")
    
    if cfg.mode == "train":
        trainset = load_data(os.path.join(cfg.data.path, "train.h5"),
                            window=window,
                            spatial=cfg.data.spatial)
        print("Train data min max", trainset['data'].min(), trainset['data'].max(), trainset['data'].shape)
        validset = load_data(os.path.join(cfg.data.path, "valid.h5"),
                            window=window,
                            spatial=cfg.data.spatial)
        print("Valid data min max", validset['data'].min(), validset['data'].max(), validset['data'].shape)
    if cfg.mode == "eval":
        test_dataset = load_data(os.path.join(cfg.data.path, "test.h5"),
                            window=window,
                            spatial=cfg.data.spatial)
        print("Test data min max",test_dataset['data'].min(), test_dataset['data'].max(), test_dataset['data'].shape)


    # Network
    print("Making the score")


    score = make_score(
        cfg.score,
        cfg.net,
        window,
        cfg.data.spatial,
        condition_dim=condition_dim,
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

    # lr = lambda t: (1 + math.cos(math.pi * t / epochs)) / 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=epochs,
        eta_min=1e-6)
    
    epoch_offset = 0

    if cfg.mode == "train":
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

    if cfg.mode == "eval":
        print("Loading the ckpt")
        training_dir = get_training_dir(current_dir, cfg)
        ckpt_path = Path(os.path.join(training_dir, cfg.ckpt_dir, "score_ema.pth"))
        score = load_score(ckpt_path)
        print('Checkpoint path', ckpt_path)

        sde = VPSDE(
            eps=score.kernel,
            shape=shape,
            model_type=cfg.model_type,
        ).cuda()
        
        window = cfg.window
        fixed_horizon = cfg.fixed_horizon
        num_observed = cfg.data.spatial * (window - cfg.predictive_horizon)

        # Loop to run the process 10 times
        for run_idx in range(3):
            print(f"Run {run_idx + 1}")

            print('Generating conditional samples')

            test_batch_size = 1

            true_x = test_dataset['data']

            print(f'True test samples size: {true_x.shape}')


            all_conditional_samples = []
            
            sde.eval()

            for i in range(1500):
                x = true_x[i].unsqueeze(0)
                #print(x.shape)
                x_condition = x.clone()

                mask = torch.zeros_like(x_condition)
                mask[:, :num_observed] = 1.0
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                x_condition = torch.cat([mask, mask*x_condition], dim=1).to(device).cuda()

                #print(x_condition.shape)
                sde.net.set_condition(x_condition)
                
                #print('set condition done')
                
                conditional_samples = sde.sample()
                #print(conditional_samples.shape)

                all_conditional_samples += [conditional_samples]

            all_conditional_samples = torch.cat(all_conditional_samples, dim=0)

            # Extract the data you want to save
            saved_data = all_conditional_samples

            # Debugging: print the shape
            print(f"[Run {run_idx + 1}] Shape of saved data: {saved_data.shape}")

            # Define the path to save the .npy file
            npy_path = f"{run_idx + 1}/condition.npy"

            # Save the data as a NumPy array
            np.save(npy_path, saved_data.cpu().numpy())

            print(f"[Run {run_idx + 1}] Conditional samples saved to: {npy_path}")


        logger.close()

if __name__ == "__main__":
    main()
