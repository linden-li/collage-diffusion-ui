import torch
from dataclasses import dataclass

from omegaconf import OmegaConf


@dataclass
class BackendConfig:
    config_name: str


@dataclass
class CompvisConfig(BackendConfig):
    omega_config: OmegaConf
    model_ckpt: str


@dataclass
class DiffusersConfig(BackendConfig):
    repo_id: str
    torch_dtype: torch.dtype
    revision: str
