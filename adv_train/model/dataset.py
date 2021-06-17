from enum import Enum
from dataclasses import dataclass
from omegaconf import MISSING


class DatasetType(Enum):
    MNIST = "mnist"
    CIFAR = "cifar"


@dataclass
class DatasetParams:
    data_dir: str = "./data"
    batch_size: int = 256
    test_batch_size: int = 512
    num_workers: int = 4
    shuffle: bool = True
    download: bool = True