import os
import subprocess
import time
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

REPO_ROOT = os.environ.get("SYNTHOLOGY_ROOT", "../../../..")


@hydra.main(version_base=None, config_path=f"{REPO_ROOT}/configs/lubm", config_name="config")
def main(cfg: DictConfig):
    t_global_start = time.perf_counter()
    # Dynamically locate the JAR file in vendor folder
    base_dir = Path(__file__).resolve().parent.parent.parent
    logger.info(f"Base LUBM directory: {base_dir}")

    jar_path = base_dir / "src" / "lubm" / "vendor" / "lubm-uba.jar"

    if not jar_path.exists():
        logger.error(f"Cannot find LUBM generator at {jar_path}")
        return

    # Base output directory
    original_cwd = Path(hydra.utils.get_original_cwd())
    output_dir_cfg = cfg.dataset.output_dir
    output_path = Path(output_dir_cfg) if Path(output_dir_cfg).is_absolute() else original_cwd / output_dir_cfg

    raw_data_dir = output_path / "raw"

    # Dataset sizes to generate
    dataset_configs = cfg.dataset.sizes

    # Auto-detect cores if threads is set to 0
    num_threads = cfg.generator.threads
    if num_threads == 0:
        num_threads = os.cpu_count() or 1
        logger.info(f"Auto-detected {num_threads} cores for generation.")

    for univ_count in dataset_configs:
        # Create a specific folder for this size (e.g., data/lubm_raw/lubm_1)
        output_folder = raw_data_dir / f"lubm_{univ_count}"
        output_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Generating LUBM-{univ_count} ---")
        print(f"Outputting to: {output_folder}")

        # Construct the Java command using configuration flags
        cmd = [
            "java",
            "-jar",
            str(jar_path),
            "--univ",
            str(univ_count),
            "--format",
            str(cfg.generator.format),
            "--consolidate",
            str(cfg.generator.consolidate),
            "--threads",
            str(num_threads),
            "--output",
            str(output_folder),
        ]

        # Execute the command
        t_ds_start = time.perf_counter()
        try:
            subprocess.run(cmd, check=True)
            elapsed = time.perf_counter() - t_ds_start
            logger.info(f"Success: LUBM-{univ_count} generated in {elapsed:.2f}s.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error generating LUBM-{univ_count}: {e}")

    logger.info(f"Total LUBM generation runtime: {time.perf_counter() - t_global_start:.2f}s")


if __name__ == "__main__":
    main()
