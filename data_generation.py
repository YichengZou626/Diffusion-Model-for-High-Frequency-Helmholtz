import jax.numpy as jnp
import numpy as np
from jax import random, jit
import os
import logging
from jwave import FourierSeries
from jwave.geometry import Domain, Medium
from jwave.acoustics.time_harmonic import helmholtz_solver

# Setup logging to write output to a file
log_file = "/work/yz886/J-wave/5e5Hz/results.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")

# Define the GRF function
def GRF2d_exponential(Nx, Ny, alpha, length):
    """Generate a 2D Gaussian Random Field (GRF)."""
    kx = np.fft.fftfreq(Nx, 1 / Nx)
    ky = np.fft.rfftfreq(Ny, 1 / Ny)
    Kx, Ky = np.meshgrid(kx, ky)
    K = np.sqrt(Kx**2 + Ky**2).T

    lmbda = jnp.array(np.exp(-(length * K) ** alpha))
    eta = jnp.array(np.random.randn(*K.shape) + 1j * np.random.randn(*K.shape))
    uhat = lmbda * eta
    u = np.fft.irfft2(uhat, s=(Nx, Ny), norm='forward')
    return jnp.array(u - jnp.mean(u))

def create_circular_source(N, radius, amplitude=1.0):
    """Create a circular source pattern at the center."""
    # If N is a tuple, take the first element
    N = N[0] if isinstance(N, tuple) else N

    x = jnp.arange(N) - N//2
    y = jnp.arange(N) - N//2
    X, Y = jnp.meshgrid(x, y)
    R = jnp.sqrt(X**2 + Y**2)

    src_field = jnp.zeros((N, N)).astype(jnp.complex64)
    src_field = src_field.at[R <= radius].set(amplitude)

    return src_field

# Solve the Helmholtz equation
@jit
def solve_helmholtz(medium, omega, src):
    return helmholtz_solver(medium, omega, src)

# Function to generate a single wavefield
def generate_wavefield(N, dx, omega):
    while True:
        alpha = np.random.uniform(1, 5)
        length = np.random.uniform(0.35, 0.6)
        grf_field = GRF2d_exponential(N[0], N[1], alpha, length)
        sound_speed = 1500 + 100 * grf_field

        if jnp.all((sound_speed > 0) & (sound_speed < 3000)):
            break

    domain = Domain(N, dx)
    sound_speed = FourierSeries(jnp.expand_dims(sound_speed, -1), domain)
    medium = Medium(domain=domain, sound_speed=sound_speed, density=1000.0, pml_size=15)

    radius = 10
    src_field = create_circular_source(N, radius)

    src = FourierSeries(jnp.expand_dims(src_field, -1), domain) * omega
    field = solve_helmholtz(medium, omega, src)

    return (
        np.array(sound_speed.on_grid.squeeze()),  
        np.array(src.on_grid.squeeze()),   
        np.array(field.on_grid.squeeze()),   
        np.array(domain.grid.squeeze()),
        alpha, length
    )

# Simulation parameters
N = (256, 256)
dx = (0.001, 0.001)
omega = 5e5

# Save directory
save_dir = "/work/yz886/J-wave/5e5Hz/"
os.makedirs(save_dir, exist_ok=True)

# Create storage arrays
num_samples = 10000
sound_maps = np.zeros((num_samples, N[0], N[1]), dtype=np.float32)
srcs = np.zeros((num_samples, N[0], N[1]), dtype=np.complex64)
fields = np.zeros((num_samples, N[0], N[1]), dtype=np.complex64)
domains = np.zeros((num_samples, N[0], N[1], 2), dtype=np.float32)

# Generate and store wavefields
for i in range(num_samples):
    sound_map, src, field, domain, alpha, length = generate_wavefield(N, dx, omega)
    sound_maps[i] = sound_map
    srcs[i] = src
    fields[i] = field
    domains[i] = domain

    # Print to Slurm output and log file
    msg = f"Generated Wavefield {i + 1}: alpha={alpha:.2f}, length={length:.3f}"
    print(msg)  # This ensures output appears in Slurm logs
    logging.info(msg)  # This saves output to results.log file

# Save final datasets
np.save(os.path.join(save_dir, "sound_maps.npy"), sound_maps)
np.save(os.path.join(save_dir, "srcs.npy"), srcs)
np.save(os.path.join(save_dir, "fields.npy"), fields)
np.save(os.path.join(save_dir, "domains.npy"), domains)

print("All data saved successfully!")
logging.info("All data saved successfully!")

