# Project Progress

This page summarizes work completed so far. The numbers below are preliminary
engineering results from the current research prototype, not a finished stock
assessment benchmark.

## Infrastructure Built

- Set up a CUDA training workstation with an RTX 3090 for larger teacher-model
  experiments.
- Built a conversion pipeline from VIAME/DIVE-style scallop annotations into
  YOLO-format datasets.
- Generated multiple dataset layouts for model comparison:
  - 640px letterboxed baseline datasets.
  - 640px tiled/mosaic edge datasets to preserve scallop detail.
  - 1280px left/right panel datasets for higher-capacity teacher models.
  - Augmented 1280px teacher datasets using flips and scale changes.
- Added scripts for composite RGB/depth frame preparation, rectangular scout
  dataset generation, and ONNX-to-RKNN conversion.

## Training Completed

| Run | Hardware | Model | Dataset Shape | Result |
| --- | --- | --- | --- | --- |
| Baseline small model | RX 6600, later RTX 3090 | YOLO11n / YOLOv5s class models | 640px public-data derivatives | Established first working scallop detector baselines. |
| Teacher round 1 | RTX 3090 | YOLO11m | Public Zenodo scallop data | Best recorded mAP50: 0.825, mAP50-95: 0.525. |
| Large teacher test | RTX 3090 | YOLO26x at 1280px | Augmented public-data derivative | 5-epoch test reached mAP50: 0.797, mAP50-95: 0.486. |
| Left/right teacher | RTX 3090 | YOLO26x at 1280px | Split-panel public-data derivative | Full run reached about mAP50: 0.654, mAP50-95: 0.403. |
| Edge mosaic student | RTX 3090 | YOLOv5s 640px, 4 classes | 640px tiled/mosaic derivative | mAP50: 0.549 overall; queen scallop class mAP50: 0.841. |

The current learning from these runs is that preserving image detail matters more
than simply shrinking full 1920x1080 frames into 640x640 inputs. Tiled/mosaic and
left/right panel datasets are more promising for small or partially buried
scallops.

## Edge Deployment On Luckfox Aura

The project has moved beyond offline training into early on-device testing:

- Exported candidate models through ONNX and RKNN conversion flows.
- Validated a 512x288 INT8 scout model on the Aura runtime.
- Validated a 640x640 INT8 model as a second-stage mosaic/zoom candidate, but
  this path was slower than desired on the Aura.
- Tested a live two-stage architecture: fast scout pass followed by higher-detail
  mosaic/zoom confirmation.
- Confirmed that the scout-to-mosaic handoff works on live camera input, while
  identifying speed and box-alignment issues that make a classifier-based second
  stage more attractive.

Known remaining work:

- Improve box scale alignment in the mosaic/zoom view.
- Move more post-processing out of Python where possible.
- Run more held-out video validation before making any operational claims.
- Add clearer public reproduction instructions for training and export.

## Next Architecture To Test

The next experiments separate fast proposal generation from confirmation:

- Server / GPU path: scout detector -> YOLO26x single-label confirmation model
  and SwinV2 classifier.
- Aura path: YOLOv5s scout detector -> MobileNetV2 classifier.

The aim is to keep the embedded device responsive while reserving heavier models
for the GPU path or compact classifier heads.

## Published Artifacts

Large artifacts are not stored in GitHub. Some derivative datasets and model
weights have been prepared or uploaded to Hugging Face so they can be shared
without bloating this source repository.

- Keep GitHub source-only: code, docs, configs, and small examples.
- Release derivative datasets and models under CC-BY-4.0 when they are derived
  from the public St Andrews / NatureScot Zenodo dataset.
- Make any artifact private or access-controlled if it includes non-public raw
  footage, private transfer links, credentials, unpublished partner data, or
  unreviewed material supplied under unclear terms.
- Do not publish additional NatureScot-provided data until the data custodian has
  confirmed sharing terms.

Before linking public Hugging Face repositories from this README, each model or
dataset card should clearly state:

- the original Zenodo DOI: https://doi.org/10.5281/zenodo.10156830;
- that the artifact is a derivative work;
- the CC-BY-4.0 license;
- what transformations were applied;
- that results are preliminary and research/prototype oriented.
