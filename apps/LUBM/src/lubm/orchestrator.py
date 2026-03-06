import subprocess
from pathlib import Path

from loguru import logger

def generate_multiple_datasets():
    # Dynamically locate the JAR file in vendor folder
    base_dir = Path(__file__).resolve().parent.parent.parent

    logger.info(f"Base directory: {base_dir}")

    jar_path = base_dir / "src" / "lubm" / "vendor" / "lubm-uba.jar"
    
    if not jar_path.exists():
        raise FileNotFoundError(f"Cannot find LUBM generator at {jar_path}")

    # Base output directory
    data_dir = base_dir.parent.parent / "data" / "lubm" / "raw"
    
    # Dataset sizes to generate
    dataset_configs = [1, 5, 10]

    for univ_count in dataset_configs:

        # Create a specific folder for this size (e.g., data/lubm_raw/lubm_1)
        output_folder = data_dir / f"lubm_{univ_count}"
        output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"\n--- Generating LUBM-{univ_count} ---")
        print(f"Outputting to: {output_folder}")

        # Construct the Java command using the best performance flags
        cmd = [
            "java", "-jar", str(jar_path),
            "--univ", str(univ_count),
            "--format", "TURTLE",
            "--consolidate", "Maximal", # Highly recommended for Turtle
            "--threads", "8",           # Adjust based on your machine/HPC
            "--output", str(output_folder)
        ]

        # Execute the command
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Success: LUBM-{univ_count} generated.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error generating LUBM-{univ_count}: {e}")

if __name__ == "__main__":
    generate_multiple_datasets()
