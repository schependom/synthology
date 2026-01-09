# Run Patrick Hohenecker's family tree generator
# with configuration from
# configs/asp_generator/family-tree.yaml

import hydra
from ftdatagen.__main__ import main
from ftdatagen.config import Config

# Read config from configs/asp_generator/family-tree.yaml and run

REPO_ROOT = "../.."


@hydra.main(version_base=None, config_path=f"{REPO_ROOT}/configs/asp_generator/", config_name="family-tree")
def run_family_tree_generator(cfg: Config):
    conf = Config()
    conf.dlv = cfg.dlv
    conf.max_branching_factor = cfg.max_branching_factor
    conf.max_tree_depth = cfg.max_tree_depth
    conf.max_tree_size = cfg.max_tree_size
    conf.negative_facts = cfg.negative_facts
    conf.num_samples = cfg.num_samples
    conf.output_dir = cfg.output_dir
    conf.quiet = cfg.quiet
    conf.seed = cfg.seed
    conf.stop_prob = cfg.stop_prob
    conf.ontology_path = cfg.ontology_path

    main(conf)


if __name__ == "__main__":
    run_family_tree_generator()
