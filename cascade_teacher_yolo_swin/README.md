---
license: cc-by-4.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
- object-detection
- image-classification
- cascade
- scallop
- underwater
- fisheries
---

# FishingROV Cascade Teacher Bundle (YOLO 1cls + SwinV2 384)

Single-folder, repeatable bundle for FishingROV cascade feedback runs.

- Detector: Chungus YOLO 1cls (`chungus_1cls_best.pt`)
- Classifier: SwinV2-B 384 clean4 (`swinv2_384_best.pt`)
- Runner: config-driven video inference (`scripts/run_video.sh`)
- Outputs: VIAME CSV + DIVE JSON + attrs + run summary

This bundle supersedes prior split usage of older 1cls detector + SwinV2-256 references for cascade testing.

## Script arguments (quick reference)

Use this section first if you are wiring the bundle into your own environment.

### `scripts/run_video.sh`

Usage:

```bash
bash scripts/run_video.sh /path/to/video.mp4 [optional_out_dir]
```

Arguments:
- `$1` (required): video file path
- `$2` (optional): output directory (defaults to `runs/video_YYYYmmdd_HHMMSS`)

Env override:
- `CONFIG=/path/to/config.json` to use a different mode config

### `scripts/run_images.sh`

Usage:

```bash
bash scripts/run_images.sh /path/to/images_dir [optional_out_dir]
```

Arguments:
- `$1` (required): folder containing images (`.jpg/.jpeg/.png/.bmp/.webp`)
- `$2` (optional): output directory

Env override:
- `CONFIG=/path/to/config.json` to use a different mode config

### `scripts/run_from_config.py`

Usage (video):

```bash
scripts/run_from_config.py --config config/mode_video_default.json --video-input /path/to/video.mp4
```

Usage (images):

```bash
scripts/run_from_config.py --config config/mode_video_default.json --images-dir /path/to/images
```

Arguments:
- `--config` (required): mode config JSON
- `--video-input` (required for video mode): input video path
- `--images-dir` (required for image mode): image-folder path
- `--out-dir` (optional): explicit output folder
- `--skip-convert` (optional flag): skip `dive convert` and `dive verify-dive-json`
- `--max-images` (optional): runtime override for max frames/images processed

Note:
- Provide exactly one of `--video-input` or `--images-dir`.

### `scripts/fetch_weights.sh`

Usage:

```bash
bash scripts/fetch_weights.sh
```

Behavior:
- Reads `weights.*.hf_url` and `weights.*.local` from config
- Downloads missing weights only

Env override:
- `CONFIG=/path/to/config.json` to fetch from a different config

## Dependencies and setup

### Python/runtime dependencies

- Python 3.11+
- `numpy`
- `torch`
- `ultralytics`
- `timm`
- `torchvision`
- `opencv-python`
- `Pillow`
- `pyyaml`

Install example:

```bash
python -m pip install numpy torch ultralytics timm torchvision opencv-python Pillow pyyaml
```

### Optional CLI dependency

- `dive` CLI is required only when conversion/verification is enabled.
- If `dive` is unavailable, run with `--skip-convert`.

### Paths you may need to change

Edit `config/mode_video_default.json`:
- `paths.python`: Python executable for your environment
- `paths.infer_script`: path to `infer_cascade_yolo_swin_to_viame.py`
- `paths.dive_bin`: dive binary name/path
- `weights.*.local`: local weight paths
- `weights.*.hf_url`: remote weight URLs

## Hardware configuration and tuning

All hardware tuning is done in `config/mode_video_default.json` under `runtime` and `video`.

Main runtime knobs:
- `runtime.conf`: detector confidence threshold (default `0.1` for reported comparison)
- `runtime.imgsz`: detector inference size (higher = more compute, often better recall)
- `runtime.yolo_batch`: detector batch size
- `runtime.cls_batch`: classifier batch size
- `runtime.max_images`: cap on total processed frames/images (`0` = no cap)
- `video.fps_sample`: sampling rate for video frame extraction

Recommended starting points by hardware class:

- Low-memory CPU or small GPU:
  - set `runtime.yolo_batch=1`
  - set `runtime.cls_batch=2` to `4`
  - reduce `runtime.imgsz` (for example `960`)
  - reduce `video.fps_sample` if throughput is low
- Mid-range GPU:
  - keep `runtime.yolo_batch=1`
  - set `runtime.cls_batch=8` to `16`
  - keep `runtime.imgsz=1280` if memory allows
- High-memory GPU:
  - increase `runtime.cls_batch` first
  - optionally raise `runtime.yolo_batch` to `2` if stable
  - keep the detector at `runtime.conf=0.1` if you want comparability with card metrics

If running on different cameras or panel geometry, update:
- `runtime.left_xmin`, `runtime.left_xmax`
- `runtime.right_xmin`, `runtime.right_xmax`
- `runtime.roi_ymin`, `runtime.roi_ymax`

If matching/merge behavior needs adjustment, update:
- `runtime.merge_iou`

If class balance or operating point differs, update:
- `runtime.class_thresholds` (format: `king=...,queen=...,dead=...,not_a_scallop=...`)

## Preliminary held-out cascade metrics

Status: **PRELIMINARY / NOT INDEPENDENTLY VERIFIED**.

These are internal station-disjoint held-out metrics from the current pipeline experiment log and are provided for early feedback, not as a final benchmark.

Held-out split: `scallop_lr_teacher_1280_v2_heldout_4cls` test subset.

Detector stage for the comparison below is run at `conf=0.1`.

Cascade decision rule used for these metrics:
- per-class weighted normalized argmax on classifier softmax,
- `argmax(softmax_prob[class] / class_threshold[class])`,
- thresholds: `king=0.1, queen=0.7, dead=0.14, not_a_scallop=1.0`.

| Metric | Value |
| --- | --- |
| King F1 | 0.541 |
| Queen F1 | 0.773 |
| Dead F1 | 0.459 |
| Macro F1 (K/Q/D) | 0.591 |

Reference source in project experiments log: EXP-2026-06-16-PIPELINE-CONF-0P1-NORMARGMAX.

## Comparison context (same held-out split)

| Source | king F1 | queen F1 | dead F1 | Macro F1 (K/Q/D) | Detector P | Detector R |
| --- | --- | --- | --- | --- | --- | --- |
| Cascade conf=0.1 + normalized weighted argmax (this card) | 0.541 | 0.773 | 0.459 | **0.591** | 0.649 | 0.660 |
| Chungus 4cls detector-only baseline | 0.392 | 0.633 | 0.364 | **0.463** | 0.602 | 0.478 |

Bundle default runtime config also uses detector `conf=0.1`.

Global classifier-threshold sweep (matched detections, same run):
- best global threshold: `0.35`
- precision: `0.852`
- recall: `0.992`
- F1: `0.917`

Interpretation caveat:
- Cascade rows and detector-only rows answer different deployment questions. Treat this as deployment-oriented internal evidence, not an external benchmark claim.

## HF weights

Bundle repo:
- https://huggingface.co/FishingROV/cascade_teacher_yolo_swin

Direct weight links:
- Detector (Chungus 1cls): https://huggingface.co/FishingROV/cascade_teacher_yolo_swin/resolve/main/weights/chungus_1cls_best.pt
- Classifier (SwinV2 384): https://huggingface.co/FishingROV/cascade_teacher_yolo_swin/resolve/main/weights/swinv2_384_best.pt

## Contents

- `config/mode_video_default.json`: default runtime mode
- `scripts/run_video.sh`: one-command entrypoint
- `scripts/run_from_config.py`: config runner
- `scripts/fetch_weights.sh`: HF weight fetch helper
- `experimental/with_viame/cascade_teacher_yolo_swin.pipe`: experimental VIAME/KWIVER pipeline scaffold
- `experimental/with_viame/README.md`: outstanding requirements for drop-in DIVE compatibility
- `weights/chungus_1cls_best.pt` (local optional cache, ignored by git)
- `weights/swinv2_384_best.pt` (local optional cache, ignored by git)

## Experimental VIAME .pipe scaffold

If your goal is DIVE Web/Desktop pipeline integration, start with:

- `experimental/with_viame/cascade_teacher_yolo_swin.pipe`
- `experimental/with_viame/README.md`

Important:
- This `.pipe` is intentionally marked experimental and is not guaranteed drop-in across all VIAME builds.
- The README in that folder lists outstanding requirements to reach true drop-in compatibility.

## Dependencies

- Python 3.11+ with:
  - `numpy`
  - `torch`
  - `ultralytics`
  - `timm`
  - `torchvision`
  - `opencv-python`
  - `Pillow`
  - `pyyaml`
- DIVE CLI (`dive`) for optional conversion/verify.

## Quick run

```bash
bash scripts/run_video.sh /path/to/video.mp4
```

Optional explicit output directory:

```bash
bash scripts/run_video.sh /path/to/video.mp4 /path/to/output_dir
```

Run on an image folder:

```bash
bash scripts/run_images.sh /path/to/images_dir
```

Optional explicit output directory (images):

```bash
bash scripts/run_images.sh /path/to/images_dir /path/to/output_dir
```

## Attribution & License

This model bundle is a derivative work based on the **University of St Andrews King Scallop dataset**.
- Original DOI: [10.5281/zenodo.10156830](https://doi.org/10.5281/zenodo.10156830)

In accordance with the original dataset's terms, this derivative work is released under the **Creative Commons Attribution 4.0 International (CC-BY 4.0)** license. You are free to share and adapt this material, provided you give appropriate credit to the original authors and indicate if changes were made.

NatureScot attribution note:
- FishingROV development context includes collaboration and field framing aligned with NatureScot fisheries/scallop-survey objectives. This bundle does not include proprietary NatureScot raw assets; it packages trained cascade weights and reproducible inference tooling derived from the attributed St Andrews source lineage that was sourced from NatureScot assets.
