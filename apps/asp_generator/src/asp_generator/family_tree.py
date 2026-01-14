# Run Patrick Hohenecker's family tree generator
# with configuration from
# configs/asp_generator/family-tree.yaml
import os

import hydra
from loguru import logger
from omegaconf import OmegaConf

from .__main__ import main
from .config import Config

# Read config from configs/asp_generator/family-tree.yaml and run
# Fallback to relative path if SYNTHOLOGY_ROOT not set (e.g. during simple package tests)
REPO_ROOT = os.environ.get("SYNTHOLOGY_ROOT", "../../../../..")


@hydra.main(version_base=None, config_path=f"{REPO_ROOT}/configs/asp_generator/", config_name="config")
def run_family_tree_generator(cfg: Config):
    conf = Config()

    logger.info(f"Running family tree ASP generator with Hydra configuration: {OmegaConf.to_yaml(cfg)}")

    conf.dlv = cfg.dlv
    conf.max_branching_factor = cfg.dataset.max_branching_factor
    conf.max_tree_depth = cfg.dataset.max_tree_depth
    conf.max_tree_size = cfg.dataset.max_tree_size
    conf.negative_facts = cfg.dataset.negative_facts
    conf.num_samples = cfg.dataset.num_samples
    conf.output_dir = cfg.dataset.output_dir
    conf.quiet = cfg.dataset.quiet
    conf.seed = cfg.dataset.seed
    conf.stop_prob = cfg.dataset.stop_prob
    conf.ontology_path = cfg.dataset.ontology_path

    main(conf)


if __name__ == "__main__":
    try:
        run_family_tree_generator()
        logger.success("Family tree `reldata` files successfully written.")
    except Exception as e:
        logger.error(f"Error running family tree generator: {e}")
        exit(1)
