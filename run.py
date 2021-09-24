import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from model import CelebaDataModule, CmnistDataModule, Experiment

cs = ConfigStore.instance()
cs.store(node=Experiment, name="Experiment")
cs.store(node=CelebaDataModule, name="CelebaDataModule", package="data", group="class/data")
cs.store(node=CmnistDataModule, name="CmnistDataModule", package="data", group="class/data")


@hydra.main(config_path="conf", config_name="config")
def main(hydra_config: DictConfig):
    exp = OmegaConf.to_object(hydra_config)
    assert isinstance(exp, Experiment)  # .to_object() turned the config object into the real object
    exp.train()


if __name__ == "__main__":
    main()
