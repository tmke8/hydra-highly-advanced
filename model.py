from abc import abstractmethod
from enum import Enum, auto
import attr

__all__ = ["CelebaDataModule", "CmnistDataModule", "Experiment"]


@attr.s(auto_attribs=True, kw_only=True, repr=False, eq=False, init=False)
class DataModule:
    def __init__(self):
        self.initialized: bool = True

    @abstractmethod
    def get_name(self) -> str:
        """Name of the dataset."""


class CelebAttr(Enum):
    gender = auto()
    hair_color = auto()


@attr.s(auto_attribs=True, kw_only=True, repr=False, eq=False)
class CelebaDataModule(DataModule):
    target_attr: CelebAttr = CelebAttr.gender

    def __attrs_pre_init__(self):
        super().__init__()

    def get_name(self) -> str:
        return "celeba"


@attr.s(auto_attribs=True, kw_only=True, repr=False, eq=False)
class CmnistDataModule(DataModule):
    padding: int = 2

    def __attrs_pre_init__(self):
        super().__init__()

    def get_name(self) -> str:
        return "cmnist"


@attr.s(auto_attribs=True, kw_only=True, repr=False, eq=False)
class Experiment:
    """Main class for the experiment."""

    data: DataModule
    seed: int = 42
    use_wandb: bool = False

    def train(self) -> None:
        print("Start training...")
        assert isinstance(self.data, DataModule)
        print(f"Dataset: {self.data.get_name()}")
        print(f"Is initialized: {self.data.initialized}")
