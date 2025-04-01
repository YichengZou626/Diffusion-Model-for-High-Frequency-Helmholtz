from .helmholtz import HelmholtzDataset, load_helmholtz
from .pt_dataset import PTDataset
from .burgers import Burgers1dTimeDataset
from .dict_dataset import DictDataset
from .mesh_datamodule import MeshDataModule
from .car_cfd_dataset import CarCFDDataset

# only import SphericalSWEDataset if torch_harmonics is built locally
try:
    from .spherical_swe import load_spherical_swe
except ModuleNotFoundError:
    pass
