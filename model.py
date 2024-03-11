from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, final
from typing_extensions import Self

from third_party import DataModule, Trainer

__all__ = ["CelebaDataModule", "CmnistDataModule", "Experiment"]


@dataclass(kw_only=True, repr=False, eq=False)
class DcDataModule(DataModule):
    """Dataclass-compatible base of our data modules."""

    @final
    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        obj = object.__new__(cls)
        # We have to ensure that `super().__init__()` is called, so that the objects
        # are properly initialized.
        DataModule.__init__(obj)
        return obj


class CelebAttr(Enum):
    GENDER = auto()
    HAIR_COLOR = auto()


@dataclass(kw_only=True, repr=False, eq=False)
class CelebaDataModule(DcDataModule):
    target_attr: CelebAttr = CelebAttr.GENDER
    hidden_dims: tuple[int, ...] = (128, 64)

    def get_name(self) -> str:
        return "celeba"


@dataclass(kw_only=True, repr=False, eq=False)
class CmnistDataModule(DcDataModule):
    padding: int = 2

    def get_name(self) -> str:
        return "cmnist"


@dataclass
class TrainerConf:
    """Config class with a target.

    When running `instantiate()`, the constructor of the specified target is called.
    """

    _target_: str = "third_party.Trainer"
    num_gpus: int = 0
    max_epochs: Optional[int] = None
    val_check_interval: float = 1.0
    precision: int = 32


@dataclass(kw_only=True)
class Experiment:
    """Main class for the experiment."""

    data: DcDataModule
    trainer: TrainerConf = field(default_factory=TrainerConf)
    seed: int
    use_wandb: bool = False

    def train(self) -> None:
        print("Start training...")
        assert isinstance(self.data, DataModule)
        print(f"Dataset: {self.data.get_name()}")
        print(f"Is initialized: {self.data.initialized}")
        assert isinstance(self.trainer, Trainer)
        print(f"trainer.num_gpus: {self.trainer.num_gpus}")
        match self.data:
            case CelebaDataModule():
                print(f"target_attr: {self.data.target_attr}")
                print(f"hidden_dims: {self.data.hidden_dims}")
                print(f"hidden_dims type: {type(self.data.hidden_dims)}")
            case CmnistDataModule():
                print(f"padding: {self.data.padding}")
            case _:
                pass
