from train import Trainer
from copy import deepcopy
import hydra
from omegaconf import DictConfig

@hydra.main(config_name='config', config_path='.')
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
