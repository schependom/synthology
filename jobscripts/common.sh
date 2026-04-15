#!/bin/sh

synthology_enter_repo() {
  if [ "${REPO_ROOT+x}" != "x" ] || [ -z "${REPO_ROOT}" ]; then
    echo "ERROR: REPO_ROOT must be set by the calling script" >&2
    return 1
  fi

  export REPO_ROOT
  cd "${REPO_ROOT}" || exit 1
}


synthology_setup_runtime_storage() {
  storage_root="${1:-${REPO_ROOT}/.cache/runtime}"

  mkdir -p \
    "${storage_root}/tmp" \
    "${storage_root}/xdg-cache" \
    "${storage_root}/xdg-config" \
    "${storage_root}/xdg-data" \
    "${storage_root}/xdg-state" \
    "${storage_root}/wandb" \
    "${storage_root}/wandb-cache" \
    "${storage_root}/wandb-artifacts"

  export TMPDIR="${storage_root}/tmp"
  export XDG_CACHE_HOME="${storage_root}/xdg-cache"
  export XDG_CONFIG_HOME="${storage_root}/xdg-config"
  export XDG_DATA_HOME="${storage_root}/xdg-data"
  export XDG_STATE_HOME="${storage_root}/xdg-state"
  export WANDB_DIR="${storage_root}/wandb"
  export WANDB_CACHE_DIR="${storage_root}/wandb-cache"
  export WANDB_ARTIFACT_DIR="${storage_root}/wandb-artifacts"

  echo "Runtime storage root: ${storage_root}"
}


synthology_load_modules() {
  if ! command -v module >/dev/null 2>&1; then
    return 0
  fi

  for module_name in "$@"; do
    module load "${module_name}" || true
  done
}


synthology_activate_python_env() {
  include_dotenv="${1:-1}"

  if [ "${include_dotenv}" = "1" ] && [ -f .env ]; then
    . ./.env
  fi

  . ./.venv/bin/activate
}


synthology_sync_deps() {
  uv sync
}


synthology_require_file() {
  required_path="$1"
  label="${2:-file}"
  if [ ! -f "${required_path}" ]; then
    echo "ERROR: missing ${label}: ${required_path}" >&2
    return 1
  fi
}
