#!/usr/bin/env python3
"""Config-driven runner for private cascade video inference bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve


def _resolve(bundle_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    if p.parent == Path('.') and shutil.which(value):
        return Path(shutil.which(value) or value)
    return (bundle_root / p).resolve()


def _download_if_missing(local_path: Path, hf_url: str, label: str) -> None:
    if local_path.exists():
        return
    if not hf_url.strip():
        raise RuntimeError(f"Missing {label} weights: {local_path}. Provide file or set hf_url in config.")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[weights] downloading {label} from {hf_url} -> {local_path}")
    urlretrieve(hf_url, str(local_path))


def _run(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _collect_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: list[Path] = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def _build_temp_panel_dataset(images: list[Path], work_root: Path) -> Path:
    ds_root = work_root / "tmp_panel_dataset"
    test_dir = ds_root / "images" / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(images):
        dst = test_dir / f"{i:06d}{src.suffix.lower()}"
        try:
            dst.symlink_to(src)
        except Exception:
            shutil.copy2(src, dst)

    yaml_path = ds_root / "dataset.yaml"
    yaml_path.write_text("path: .\ntrain: images/test\nval: images/test\ntest: images/test\n", encoding="utf-8")
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cascade inference from a mode config")
    parser.add_argument("--config", required=True, help="Path to mode config JSON")
    parser.add_argument("--video-input", default="", help="Absolute or relative path to input video")
    parser.add_argument("--images-dir", default="", help="Directory of images (*.jpg, *.png, ...) for panel-style run")
    parser.add_argument("--out-dir", default="", help="Optional explicit output directory")
    parser.add_argument("--skip-convert", action="store_true", help="Skip dive convert + verify")
    parser.add_argument("--max-images", type=int, default=None, help="Override max_images from config")
    args = parser.parse_args()

    if bool(args.video_input) == bool(args.images_dir):
        raise RuntimeError("Provide exactly one of --video-input or --images-dir")

    config_path = Path(args.config).resolve()
    bundle_root = config_path.parents[1]
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    paths = cfg.get("paths", {})
    runtime = cfg.get("runtime", {})
    video_cfg = cfg.get("video", {})
    weights = cfg.get("weights", {})

    py_exe = _resolve(bundle_root, paths.get("python", "python3"))
    infer_script = _resolve(bundle_root, paths["infer_script"])
    dive_bin = paths.get("dive_bin", "dive")

    yolo_local = _resolve(bundle_root, weights["yolo"]["local"])
    swin_local = _resolve(bundle_root, weights["swin"]["local"])
    _download_if_missing(yolo_local, weights["yolo"].get("hf_url", ""), "yolo")
    _download_if_missing(swin_local, weights["swin"].get("hf_url", ""), "swin")

    if not py_exe.exists():
        raise RuntimeError(f"Python executable not found: {py_exe}")
    if not infer_script.exists():
        raise RuntimeError(f"Inference script not found: {infer_script}")

    video_input: Path | None = None
    images_dir: Path | None = None
    if args.video_input:
        video_input = Path(args.video_input)
        if not video_input.is_absolute():
            video_input = (Path.cwd() / video_input).resolve()
        if not video_input.exists():
            raise RuntimeError(f"Video input not found: {video_input}")
    else:
        images_dir = Path(args.images_dir)
        if not images_dir.is_absolute():
            images_dir = (Path.cwd() / images_dir).resolve()
        if not images_dir.is_dir():
            raise RuntimeError(f"Images directory not found: {images_dir}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (bundle_root / "runs" / f"video_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_viame = out_dir / "cascade_predictions.viame.csv"
    out_json = out_dir / "cascade_predictions.dive.json"
    out_attrs = out_dir / "cascade_predictions.attrs.json"
    out_summary = out_dir / "cascade_summary.json"

    max_images = args.max_images if args.max_images is not None else int(runtime.get("max_images", 0))

    infer_cmd = [
        str(py_exe),
        str(infer_script),
        "--yolo-weights",
        str(yolo_local),
        "--swin-weights",
        str(swin_local),
        "--conf",
        str(float(runtime.get("conf", 0.2))),
        "--imgsz",
        str(int(runtime.get("imgsz", 1280))),
        "--yolo-batch",
        str(int(runtime.get("yolo_batch", 1))),
        "--cls-batch",
        str(int(runtime.get("cls_batch", 8))),
        "--max-images",
        str(max_images),
        "--class-thresholds",
        str(runtime.get("class_thresholds", "")),
        "--left-xmin",
        str(int(runtime.get("left_xmin", 0))),
        "--left-xmax",
        str(int(runtime.get("left_xmax", 1080))),
        "--right-xmin",
        str(int(runtime.get("right_xmin", 840))),
        "--right-xmax",
        str(int(runtime.get("right_xmax", 1920))),
        "--roi-ymin",
        str(int(runtime.get("roi_ymin", 0))),
        "--roi-ymax",
        str(int(runtime.get("roi_ymax", 1080))),
        "--merge-iou",
        str(float(runtime.get("merge_iou", 0.45))),
        "--out-viame-csv",
        str(out_viame),
        "--out-summary-json",
        str(out_summary),
    ]

    if video_input is not None:
        infer_cmd.extend(
            [
                "--input-mode",
                "video",
                "--video-input",
                str(video_input),
                "--video-start-sec",
                str(float(video_cfg.get("start_sec", 0.0))),
                "--video-seconds",
                str(float(video_cfg.get("seconds", 0.0))),
                "--video-fps-sample",
                str(float(video_cfg.get("fps_sample", 1.0))),
            ]
        )
        _run(infer_cmd)
    else:
        assert images_dir is not None
        image_list = _collect_images(images_dir)
        if not image_list:
            raise RuntimeError(f"No supported images found in: {images_dir}")

        with tempfile.TemporaryDirectory(prefix="cascade_imgs_") as tmp:
            ds_yaml = _build_temp_panel_dataset(image_list, Path(tmp))
            panel_cmd = infer_cmd + [
                "--input-mode",
                "panel",
                "--panel-data-yaml",
                str(ds_yaml),
                "--panel-split",
                "test",
            ]
            _run(panel_cmd)

    if not args.skip_convert:
        if shutil.which(dive_bin) is None:
            raise RuntimeError(f"dive binary not found on PATH: {dive_bin}")
        _run([dive_bin, "convert", "viame2dive", str(out_viame), "--output", str(out_json), "--output-attrs", str(out_attrs)])
        _run([dive_bin, "verify-dive-json", str(out_json)])

    print("[done] out_dir:", out_dir)
    print("[done] summary:", out_summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
