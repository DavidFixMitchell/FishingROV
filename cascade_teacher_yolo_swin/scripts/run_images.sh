#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${BUNDLE_DIR}/config/mode_video_default.json}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /abs/or/rel/images_dir [optional_out_dir]"
  exit 2
fi

IMAGES_DIR="$1"
OUT_DIR="${2:-}"

if [[ -n "${OUT_DIR}" ]]; then
  "${BUNDLE_DIR}/scripts/run_from_config.py" --config "${CONFIG}" --images-dir "${IMAGES_DIR}" --out-dir "${OUT_DIR}"
else
  "${BUNDLE_DIR}/scripts/run_from_config.py" --config "${CONFIG}" --images-dir "${IMAGES_DIR}"
fi
