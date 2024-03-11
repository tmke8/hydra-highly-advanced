import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig

from model import CelebaDataModule, CmnistDataModule, Experiment

cs = ConfigStore.instance()
# The name of the main config has to appear in `conf/config.yaml`.
cs.store(node=Experiment, name="main_config")
# variants
cs.store(node=CelebaDataModule, name="celeba", group="data")
cs.store(node=CmnistDataModule, name="cmnist", group="data")


SUB_DIRECTORY_WITH_CONFIG_FILES = "conf"
NAME_OF_MAIN_CONFIG_FILE = "config"  # without the .yaml extension


@hydra.main(
    config_path=SUB_DIRECTORY_WITH_CONFIG_FILES,
    config_name=NAME_OF_MAIN_CONFIG_FILE,
    version_base="1.3",
)
def main(hydra_config: DictConfig):
    # Instantiate our main config object.
    # (`_convert_="object"` turns the config object into the real object.)
    exp = instantiate(hydra_config, _convert_="object")
    assert isinstance(exp, Experiment)
    exp.train()


if __name__ == "__main__":
    main()
