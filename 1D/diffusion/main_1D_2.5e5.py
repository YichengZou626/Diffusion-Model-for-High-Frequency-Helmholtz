import torch
import os
os.environ["WANDB_MODE"] = "offline"
import hydra
import math
from utils.data_processing import load_data
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

    window = 3

    condition_dim = 3

    # Data
    print("Loading data")
    
    if cfg.mode == "train":
        trainset = load_data("/work-old/yz886/ANFWI_HNO/data/1D/2.5e5/train.h5")
        trainset['data'] = trainset['data'][:10000]
        print("Train data min max", trainset['data'].min(), trainset['data'].max(), trainset['data'].shape)

        validset = load_data("/work-old/yz886/ANFWI_HNO/data/1D/2.5e5/valid.h5")
        print("Valid data min max", validset['data'].min(), validset['data'].max(), validset['data'].shape)
    if cfg.mode == "eval":
        test_dataset = load_data("/work-old/yz886/ANFWI_HNO/data/1D/2.5e5/perturbation.h5")
        print("Test data min max",test_dataset['data'].min(), test_dataset['data'].max(), test_dataset['data'].shape)


    # Network
    print("Making the score")


    score = make_score(
        cfg.score,
        cfg.net,
        window,
        condition_dim=condition_dim,
    )

    shape = (1, 256)
    
    print("gogo")

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
    
    print("Start training")

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
        
        best_valid_loss = float('inf')
        best_model_state = None  # To hold the best model weights

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
                ckpt_filename =  f"ckpt_path_{(i + 1) // 1}.pth"
                ckpt_path_latest = Path(os.path.join(current_dir, cfg.ckpt_dir, ckpt_filename))
                '''
                checkpoint = {'epoch': i + 1 + epoch_offset,
                            'model': sde.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'ema': ema.state_dict()}
                torch.save(checkpoint, ckpt_path_latest)
                '''
                with ema.average_parameters(score.parameters()):
                    torch.save(score.state_dict(), ckpt_path_latest)
                    
            # Update best model if this is the best validation loss so far
            if loss_valid < best_valid_loss:
                best_valid_loss = loss_valid
                with ema.average_parameters(score.parameters()):
                    best_model_state = score.state_dict()

        # Plot final loss curves using iterations instead of epochs
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, train_losses, label='Training Loss')
        plt.plot(iterations, valid_losses, label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.legend()
        plt.savefig('overall_loss_plot.png')
        plt.close()

        # Save
        print(ckpt_path)
        #torch.save(score.state_dict(), ckpt_path)
        with ema.average_parameters(score.parameters()):
            torch.save(score.state_dict(), ckpt_path_ema)

    if cfg.mode == "eval":
        print("Loading the ckpt")
        training_dir = get_training_dir(current_dir, cfg)
        ckpt_path = Path(os.path.join(training_dir, cfg.ckpt_dir, "score_ema_100k.pth"))
        score = load_score(ckpt_path)
        print('Checkpoint path', ckpt_path)
        
        #initial_guess = np.load("/work-old/yz886/unet_no_mid/unet_results_1D_1.5e5/guess.npy")
        #initial_guess = torch.from_numpy(initial_guess).float()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        sde = VPSDE(
            eps=score.kernel,
            shape=shape,
            model_type=cfg.model_type,
        ).cuda()
        
        window = cfg.window
        fixed_horizon = cfg.fixed_horizon
        num_observed = window - cfg.predictive_horizon
        
        total_results = []

        for run_idx in range(1):
            print(f"Run {run_idx + 1}")
            all_conditional_samples = []

            sde.eval()

            for i in range(10000):
                x = test_dataset['data'][i].unsqueeze(0)  # shape: (1, C, 256, 256)
                x_condition = x[:, :-1, :].to(device)
                
                #initial_x = initial_guess[i].unsqueeze(1).to(device)
                
                conditional_samples = sde.sample(x_condition, steps=1000)  # shape: (1, 1, 256, 256)
                all_conditional_samples.append(conditional_samples) #.squeeze(0))  # shape: (1, 256, 256)

            all_conditional_samples = torch.cat(all_conditional_samples, dim=0)  # (10, 1, 256, 256)
            total_results.append(all_conditional_samples)

        # Final stack: (3, 10, 1, 256, 256)
        saved_data = torch.stack(total_results, dim=0)

        # Remove singleton channel if needed
        saved_data = saved_data.squeeze(2)  # (3, 10, 256, 256)

        print(f"Shape of saved data: {saved_data.shape}")  # should be (3, 10, 256, 256)

        # Define the path to save the .npy file
        npy_path = "perturbation100k.npy"
        #npy_path = f"condition_{run_idx+1}.npy"

        # Save the data as a NumPy array
        np.save(npy_path, saved_data.cpu().numpy())

        print(f"Conditional samples saved to: {npy_path}")


        logger.close()

if __name__ == "__main__":
    main()
