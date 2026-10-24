import hydra
from src.trainer.trainer import Trainer


@hydra.main(version_base=None, config_path="src/configs", config_name="first_train")
def main(cfg):
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
