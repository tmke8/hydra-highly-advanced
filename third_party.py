"""This file is supposed to represent a third-party library over which we don't have control."""
from abc import abstractmethod


class DataModule:
    def __init__(self):
        self.initialized: bool = True

    @abstractmethod
    def get_name(self) -> str:
        """Name of the dataset."""
