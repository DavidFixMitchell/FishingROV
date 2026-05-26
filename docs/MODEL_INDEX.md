# Model Index

This is a sanitized public summary of models trained or tested for the
FishingROV prototype. It intentionally excludes private paths, device commands,
credentials, and large artifact locations.

Results are preliminary engineering measurements on public-data derivatives and
prototype validation splits. They should not be read as a final ecological or
stock-assessment benchmark.

## Training Runs

| Model | Role | Input / Dataset Style | Status | Preliminary Result |
| --- | --- | --- | --- | --- |
| YOLOv5s 640 | Edge baseline | 640px letterboxed public-data derivative | Trained | mAP50: 0.841 on baseline validation split. |
| YOLOv5s mosaic 640 | Edge mosaic detector | 640px tiled/mosaic public-data derivative | Trained | mAP50: 0.892 on one-class mosaic validation split. |
| YOLOv5s mosaic 640, 4-class | Edge taxonomic/condition detector | 640px tiled/mosaic derivative with dead/recessed/queen/king labels | Trained and exported for artifact hosting | mAP50: 0.549 overall; queen class mAP50: 0.841; king class remains weak. |
| YOLOv5s rectangular scout | Fast first-stage detector | 512x288 scout dataset | Trained and converted to RKNN | mAP50: 0.351; scallop-class mAP50: 0.506 (98 epochs, early stop at 68). Used as the current lightweight scout candidate on Aura. |
| YOLO26x 1280 | High-capacity teacher | 1280px augmented public-data derivative | Short test run | 5 epochs reached mAP50: 0.797 and mAP50-95: 0.486. |
| YOLO26x left/right 1280 | High-capacity teacher | 1280px left/right split panels | Full training run | Reached about mAP50: 0.654 and mAP50-95: 0.403. |

## Edge Deployment Status

| Model Family | Target | Status | Notes |
| --- | --- | --- | --- |
| 512x288 scout RKNN | Luckfox Aura / RV1126B | Working prototype | Stable enough for current first-stage scout testing. |
| 640x640 mosaic RKNN | Luckfox Aura / RV1126B | Working but slow for the intended use | Useful validation path, but no longer the preferred next architecture. |
| Legacy 640 RKNN candidates | Luckfox Aura / RV1126B | Mixed results | Some converted models loaded but produced no useful detections; invalid exports were discarded. |

## Current Direction

The next tests shift away from running a heavier 640px mosaic detector on the
Aura. The planned architecture is:

- Server / GPU path: scout detector followed by higher-capacity confirmation
  using a YOLO26x single-label model and a SwinV2 classifier.
- Aura path: YOLOv5s scout detector followed by a MobileNetV2 classifier for
  lightweight confirmation.

This keeps the embedded system focused on fast proposal generation while moving
more expensive classification to either a compact mobile classifier or the GPU
server, depending on where the video is processed.

## Artifact Policy

Model weights and derivative datasets are not stored in this Git repository.
Shareable public-data-derived artifacts may be hosted on Hugging Face with
CC-BY-4.0 attribution to the Zenodo source dataset. Any artifact containing
non-public footage or unclear sharing terms should remain private or
access-controlled.
