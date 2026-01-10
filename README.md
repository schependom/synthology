# Synthology

```
synthology/
├── apps/
│   ├── asp_generator/ # adapted from the RRN author
│   ├── ont_generator/ # the goal of my thesis
│   ├── RRN/           # created myself based on RRN paper
│   └── TransE/        # baseline model
│
├── .devcontainer/          # development container configuration
│   ├── devcontainer.json
│   └── post_create.sh
│
├── .github/                        # GitHub configuration files
│   ├── workflows/
│   │   ├── linting.yaml
│   │   ├── pre-commit-update.yaml
│   │   └── tests.yaml
│   └── dependabot.yaml
│
├── configs/                        # all Hydra configuration files
│   ├── asp_generator/
│   │   └── family-tree.yaml
│   ├── ont_generator/
│   │   └── config.yaml
│   └── rrn/
│       ├── data/
│       │   ├── dataset/
│       │   │   ├── asp.yaml
│       │   │   └── ont.yaml
│       │   └── default.yaml
│       ├── hyperparams/
│       │   └── default.yaml
│       ├── model/
│       │   └── default.yaml
│       └── config.yaml
│
├── data/
│   ├── asp/                     # input data for asp_generator
│   │   ├──  out-reldata/
│   │   ├──  out-csv/
│   │   └──  family-tree.asp
│   └── ont/                     # OWL 2 RL input data for ont_generator (thesis)
│       ├──  out-csv/
│       └──  family.ttl
│
├── models/
│   └── ...                 # trained models
│
├── checkpoints/
│   └── ...                 # trained models
│
├── src/
│   └── synthology/         # shared code across packages
│       ├── __init__.py
│       └── data_structures.py
│
├── tests/                  # unit tests
│   ├── __init__.py
│   ├── test_datagen.py
│   ├── test_rrn.py
│   └── test_transe.py
│
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── LICENSE
├── pyproject.toml
├── README.md
├── tasks.py
└── uv.lock
```
