import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig

from model import CelebaDataModule, CmnistDataModule, Experiment

cs = ConfigStore.instance()
cs.store(node=Experiment, name="Experiment")
# variants
cs.store(node=CelebaDataModule, name="CelebaDataModule", group="data")
cs.store(node=CmnistDataModule, name="CmnistDataModule", group="data")


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(hydra_config: DictConfig):
    exp = instantiate(hydra_config, _convert_="object")
    # `_convert_="object"` turns the config object into the real object
    assert isinstance(exp, Experiment)
    exp.train()


if __name__ == "__main__":
    main()
