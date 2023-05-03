from enum import Enum, auto
from typing import Optional

from attrs import define, field

from third_party import DataModule, Trainer

__all__ = ["CelebaDataModule", "CmnistDataModule", "Experiment"]


@define(kw_only=True, repr=False, eq=False)
class BaseDataModule(DataModule):
    """Base for all our own data modules."""

    def __attrs_pre_init__(self):
        # we have to manually call super().__init__() because the parent class is not an attr class
        super().__init__()


class CelebAttr(Enum):
    GENDER = auto()
    HAIR_COLOR = auto()


@define(kw_only=True, repr=False, eq=False)
class CelebaDataModule(BaseDataModule):
    target_attr: CelebAttr = CelebAttr.GENDER

    def get_name(self) -> str:
        return "celeba"


@define(kw_only=True, repr=False, eq=False)
class CmnistDataModule(BaseDataModule):
    padding: int = 2

    def get_name(self) -> str:
        return "cmnist"


@define(kw_only=True, eq=False)
class TrainerConf:
    _target_: str = "third_party.Trainer"
    num_gpus: int
    max_epochs: Optional[int] = None
    val_check_interval: float = 1.0
    precision: int = 32


@define(kw_only=True, eq=False)
class Experiment:
    """Main class for the experiment."""

    data: BaseDataModule
    trainer: TrainerConf = field(default=TrainerConf)
    seed: int = 42
    use_wandb: bool = False

    def train(self) -> None:
        print("Start training...")
        assert isinstance(self.data, DataModule)
        print(f"Dataset: {self.data.get_name()}")
        print(f"Is initialized: {self.data.initialized}")
        assert isinstance(self.trainer, Trainer)
        print(f"trainer.num_gpus: {self.trainer.num_gpus}")
        if isinstance(self.data, CelebaDataModule):
            print(f"target_attr: {self.data.target_attr}")
