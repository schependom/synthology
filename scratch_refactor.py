import os

configs_dir = r"c:\Users\home\Documents\synthology\configs\rrn"

modifications = {
    "exp1_constrained_hpc.yaml": """defaults:
  - config
  - override data/dataset: exp1_constrained
  - _self_

hyperparams:
  learning_rate: 0.0001

callbacks:
  early_stopping:
    patience: 200
  model_checkpoint:
    filename: "best-checkpoint-constrained"

logger:
  name: "exp1_constrained_hpc"
  group: "exp1_negative_sampling"
  tags: [exp1, constrained, hpc]
""",
    "exp1_proof_based_hpc.yaml": """defaults:
  - config
  - override data/dataset: exp1_proof_based
  - _self_

model:
  embedding_size: 200
  num_hidden_layers: 2

callbacks:
  early_stopping:
    patience: 200
  model_checkpoint:
    filename: "best-checkpoint-proof_based"

logger:
  name: "exp1_proof_based_hpc"
  group: "exp1_negative_sampling"
  tags: [exp1, proof_based, hpc]
""",
    "exp1_random_hpc.yaml": """defaults:
  - config
  - override data/dataset: exp1_random
  - _self_

callbacks:
  model_checkpoint:
    filename: "best-checkpoint-random"

logger:
  name: "exp1_random_hpc"
  group: "exp1_negative_sampling"
  tags: [exp1, random, hpc]
""",
    "exp2_baseline_hpc.yaml": """defaults:
  - config
  - override data/dataset: exp2_baseline
  - _self_

callbacks:
  model_checkpoint:
    filename: "best-checkpoint-exp2-baseline"

logger:
  name: "exp2_baseline_hpc"
  group: "exp2_multihop"
  tags: [exp2, baseline, hpc]
""",
    "exp2_synthology_hpc.yaml": """defaults:
  - config
  - override data/dataset: exp2_synthology
  - _self_

callbacks:
  model_checkpoint:
    filename: "best-checkpoint-exp2-synthology"

logger:
  name: "exp2_synthology_hpc"
  group: "exp2_multihop"
  tags: [exp2, synthology, hpc]
""",
    "exp3_owl2bench_hpc.yaml": """defaults:
  - config
  - override data/dataset: owl2bench
  - _self_

callbacks:
  model_checkpoint:
    filename: "best-checkpoint-exp3-owl2bench"

logger:
  name: "exp3_owl2bench_hpc"
  group: "exp3_scaling"
  tags: [exp3, owl2bench, hpc]
""",
    "exp1_constrained_hpc_test.yaml": """defaults:
  - exp1_constrained_hpc
  - _self_

test:
  checkpoint_glob: "${log_dir}/checkpoints/best-checkpoint-constrained*.ckpt"

logger:
  log_model: false
  name: "exp1_constrained_hpc_test"
  tags: [exp1, constrained, hpc, test]
""",
    "exp1_proof_based_hpc_test.yaml": """defaults:
  - exp1_proof_based_hpc
  - _self_

test:
  checkpoint_glob: "${log_dir}/checkpoints/best-checkpoint-proof_based*.ckpt"

logger:
  log_model: false
  name: "exp1_proof_based_hpc_test"
  tags: [exp1, proof_based, hpc, test]
""",
    "exp1_random_hpc_test.yaml": """defaults:
  - exp1_random_hpc
  - _self_

test:
  checkpoint_glob: "${log_dir}/checkpoints/best-checkpoint-random*.ckpt"

logger:
  log_model: false
  name: "exp1_random_hpc_test"
  tags: [exp1, random, hpc, test]
"""
}

for filename, content in modifications.items():
    filepath = os.path.join(configs_dir, filename)
    with open(filepath, "w") as f:
        f.write(content)
    print(f"Refactored {filename}")
