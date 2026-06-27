#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${BUNDLE_DIR}/config/mode_video_default.json}"

python3 - <<'PY' "${CONFIG}" "${BUNDLE_DIR}"
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])

for key in ("yolo", "swin"):
    w = cfg["weights"][key]
    local = Path(w["local"])
    if not local.is_absolute():
        local = (root / local).resolve()
    if local.exists():
        print(f"[ok] {key} exists: {local}")
        continue
    url = w.get("hf_url", "").strip()
    if not url:
        raise SystemExit(f"Missing hf_url for {key} in {sys.argv[1]}")
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {key}: {url} -> {local}")
    urlretrieve(url, str(local))

print("[done] weights ready")
PY
