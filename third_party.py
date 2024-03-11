"""This file is uupposed to represent a third-party library over which we don't have control."""

from abc import abstractmethod
from typing import Optional

__all__ = ["DataModule", "Trainer"]


class DataModule:
    def __init__(self):
        # Simulate the initialization of the data module.
        self.initialized: bool = True

    @abstractmethod
    def get_name(self) -> str:
        """Name of the dataset."""


class Trainer:
    def __init__(
        self,
        num_gpus: int,
        max_epochs: Optional[int] = None,
        val_check_interval: float = 1.0,
        precision: int = 32,
    ):
        self.num_gpus = num_gpus
        self.max_epochs = max_epochs
        self.val_check_interval = val_check_interval
        self.precision = precision
