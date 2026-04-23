import importlib
import random

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


RUNNERS = {
    "transformer": "runners.transformer",
    "kmer_logreg": "runners.kmer_logreg",
}


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    model_type = str(cfg.model.type)
    if model_type not in RUNNERS:
        raise ValueError(
            f"Unknown model.type={model_type!r}. Choose from: {list(RUNNERS)}"
        )


    runner = importlib.import_module(RUNNERS[model_type])
    runner.run(cfg)


if __name__ == "__main__":
    main()
