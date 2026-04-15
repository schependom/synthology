#!/usr/bin/env bash

set -euo pipefail

# Exp2 sweep over target caps and seeds.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
. "${REPO_ROOT}/jobscripts/common.sh"

synthology_enter_repo
synthology_setup_runtime_storage

synthology_activate_python_env 0
uv run invoke exp2-sweep-targetcaps-seeds

echo
echo "Exp2 target-cap + seed sweep complete."
