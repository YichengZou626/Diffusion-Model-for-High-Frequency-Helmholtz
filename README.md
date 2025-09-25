# Diffusion-Model-for-High-Frequency-Helmholtz

This project explores the use of conditional diffusion models to simulate high-frequency solutions to the Helmholtz equation. It focuses on learning wave propagation behavior at various frequencies using synthetic data generated via the [JWave tutorial](https://github.com/jwave-sim/jwave).

---

## Dataset

We simulate Helmholtz wavefields at the following frequencies:

- 1.5e5 Hz  
- 2.5e5 Hz  
- 5e5 Hz  
- 1.5e6 Hz  
- 2.5e6 Hz  
- 5e6 Hz  

Data generation is implemented in `data_generation.py` is adapted from the JWave framework: https://ucl-bug.github.io/jwave/notebooks/harmonic/helmholtz_problem.html.

---

## Acknowledgement

The core diffusion model implementation is based on the paper:  
**"On conditional diffusion models for PDE simulations"**

---

## Installation and Setup

To use the code for diffusion model and reproduce results, you first need to create the environment and install all the dependencies by running the following commands:

```
# Create conda environment
cd pdediff
conda env create -f environment.yml

# Activate the environment
conda activate pdediff

# Install package in editable mode
pip install -e .
```

## Experiments

We used `hydra` to manage config files and hyperparameters. All the experiment configs and hyperparameters used to train the model can be found inside the `config` folder. 

Cite for [Score-based Data Assimilation repo](https://github.com/francois-rozet/sda) and [PDERefiner repo](https://github.com/pdearena/pdearena). 

### Training a joint model on helmholtz dataset
If you want to train a model, you can run the following command

```
python main.py -m seed=0 mode=train experiment=helmholtz_1.5e6 window=3
```
where `window_size=2k+1`.

### Conditional sampling on helmholtz dataset
If you want to sample from the trained model

```
python main.py -m seed=0 mode=eval experiment=helmholtz_1.5e6 window=3
```

### Running Baseline Models
We also compare our diffusion model to other methods:
- **UNet**: with the same structure as our conditional diffusion model (SDA-based)
- **FNO (Fourier Neural Operator)**: [GitHub - neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)
- **HNO (Helmholtz Neural Operator)**: [GitHub - caifeng-zou/ANFWI_HNO](https://github.com/caifeng-zou/ANFWI_HNO)

To replicate our experiments with the FNO and HNO models:

FNO (Fourier Neural Operator)
Navigate to the script/ subfolder:
```
cd FNO/script
```
Run the training script:
```
python train_helmholtz.py
```

HNO (Helmholtz Neural Operator)
Navigate to the code/ subfolder:
```
cd HNO/code
```
Run the main script:
```
python run.py
```

### Notes
The datasets are too large to be included in this repository. To generate your own, please refer to data_generation.py.

Our goal is to assess the ability of diffusion models to simulate high-frequency Helmholtz fields, where deterministic models often fail due to frequency aliasing and lack of generalization.

## Contact
If you have any questions or suggestions, feel free to open an issue or reach out!







