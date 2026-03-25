import os
import subprocess
from pathlib import Path

from loguru import logger


def run_owl2bench_generator(
    vendor_dir: Path, profile: str, universities: int, seed: int, maven_executable: str
) -> Path:
    cmd = [
        maven_executable,
        "-q",
        "-DskipTests",
        "compile",
        "exec:java",
        "-Dexec.mainClass=ABoxGen.InstanceGenerator.Generator",
        f"-Dexec.args={universities} {profile} {seed}",
    ]

    logger.info("Running OWL2Bench Java generator: {}", " ".join(cmd))
    env = dict(os.environ)
    extra_open = "--add-opens=java.base/java.lang=ALL-UNNAMED"
    current_opts = env.get("MAVEN_OPTS", "").strip()
    env["MAVEN_OPTS"] = f"{current_opts} {extra_open}".strip()

    proc = subprocess.run(cmd, cwd=vendor_dir, text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"OWL2Bench generator failed\nExit code: {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    generated_name = f"OWL2{profile}-{universities}.owl"
    generated_path = vendor_dir / generated_name
    if not generated_path.exists():
        raise FileNotFoundError(f"Expected generated ontology not found at {generated_path}")

    logger.info("OWL2Bench generator output: {}", generated_path)
    return generated_path
