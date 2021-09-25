from enum import Enum, auto
import attr

from third_party import DataModule

__all__ = ["CelebaDataModule", "CmnistDataModule", "Experiment"]


@attr.s(auto_attribs=True, kw_only=True, repr=False, eq=False)
class BaseDataModule(DataModule):
    """Base for all our own data modules."""

    def __attrs_pre_init__(self):
        # we have to manually call super().__init__() because the parent class is not an attr class
        super().__init__()


class CelebAttr(Enum):
    gender = auto()
    hair_color = auto()


@attr.s(auto_attribs=True, kw_only=True, repr=False, eq=False)
class CelebaDataModule(BaseDataModule):
    target_attr: CelebAttr = CelebAttr.gender

    def get_name(self) -> str:
        return "celeba"


@attr.s(auto_attribs=True, kw_only=True, repr=False, eq=False)
class CmnistDataModule(BaseDataModule):
    padding: int = 2

    def get_name(self) -> str:
        return "cmnist"


@attr.s(auto_attribs=True, kw_only=True, repr=False, eq=False)
class Experiment:
    """Main class for the experiment."""

    data: BaseDataModule
    seed: int = 42
    use_wandb: bool = False

    def train(self) -> None:
        print("Start training...")
        assert isinstance(self.data, DataModule)
        print(f"Dataset: {self.data.get_name()}")
        print(f"Is initialized: {self.data.initialized}")
