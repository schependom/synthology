#!/bin/zsh
set -euo pipefail

for f in $(find configs -type f \( -name '*.yaml' -o -name '*.yml' \) | sort); do
  rel=${f#configs/}
  stem=${f##*/}
  stem=${stem%.yaml}
  stem=${stem%.yml}
  pretty=$(echo "$stem" | tr '[:lower:]' '[:upper:]' | tr '_' ' ')

  desc="CONFIGURATION FOR $pretty"
  case "$rel" in
    */dataset/*) desc="DATASET CONFIGURATION FOR $pretty" ;;
    */hyperparams/*) desc="HYPERPARAMETER CONFIGURATION FOR $pretty" ;;
    */model/*) desc="MODEL CONFIGURATION FOR $pretty" ;;
    sweep_sample.yaml) desc="WANDB SWEEP CONFIGURATION FOR JOINT EXPERIMENT TUNING" ;;
    ont_generator/config.yaml) desc="BASE CONFIGURATION FOR ONTOLOGY DATASET GENERATION" ;;
    rrn/config.yaml) desc="BASE CONFIGURATION FOR RRN TRAINING AND EVALUATION" ;;
    owl2bench/config.yaml) desc="BASE CONFIGURATION FOR OWL2BENCH GENERATION PIPELINE" ;;
  esac

  first=$(sed -n '1p' "$f")
  if [ "$first" = "#################################################################" ]; then
    tail -n +4 "$f" > "$f.__body"
    mv "$f.__body" "$f"
  fi

  {
    printf "#################################################################\n"
    printf "# %s\n" "$desc"
    printf "#################################################################\n"
    cat "$f"
  } > "$f.__new"
  mv "$f.__new" "$f"
done

echo "Headers applied to all config files under configs/."
