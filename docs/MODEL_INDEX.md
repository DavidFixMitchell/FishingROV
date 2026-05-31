# Model Index

This is a sanitized public summary of models trained or tested for the
FishingROV prototype. It intentionally excludes private paths, device commands,
credentials, and large artifact locations.

Results are engineering measurements on public-data derivatives. Headline
numbers below are on a **station-disjoint held-out set** (the public Zenodo
"Test files" survey stations, never seen in training) wherever that split
exists. They should not be read as a final ecological or stock-assessment
benchmark.

> **Method note on data leakage.** Earlier internal runs reported higher scores
> (~0.82-0.84 mAP50) using a *random frame split* of continuous survey video.
> Because neighbouring video frames are near-identical, that split leaks training
> content into validation and inflates the score. The full-frame numbers below
> have been re-measured on a **station-disjoint** split (training and evaluation
> never share a survey location). We prefer the honest, harder number over a
> flattering one built on a flawed split.

## Training Runs

| Model | Role | Input / Dataset Style | Status | Held-out Result |
| --- | --- | --- | --- | --- |
| YOLOv5s 640 | Edge baseline | 640px letterboxed public-data derivative | Trained | mAP50: **0.54** on station-disjoint held-out set (was 0.84 on a leaky random-frame split). |
| YOLOv5s mosaic 640 | Edge mosaic detector | 640px mosaic tiles | Trained | mAP50: **0.89** on a station-disjoint split, but on *synthetic mosaic tiles* (clean, centred crops) — best-case, not comparable to full-frame detectors. |
| YOLOv5s mosaic 640, 4-class | Edge taxonomic/condition detector | 640px mosaic-tile derivative with dead/recessed/queen/king labels | Trained and exported for artifact hosting | mAP50: 0.549 overall (synthetic mosaic split); queen class 0.841; king class remains weak. |
| YOLOv5s rectangular scout | Fast first-stage detector | 512x288 scout dataset | Trained and converted to RKNN | mAP50: 0.351; scallop-class mAP50: 0.506 (98 epochs, early stop at 68). Current lightweight scout candidate on Aura. |
| YOLO26x left/right 1280 | High-capacity teacher | 1280px left/right split panels | Full training run | mAP50: **0.66**, mAP50-95: 0.40 on a station-disjoint held-out split (honest). |
| YOLO26x left/right 1280, augmented | High-capacity teacher | 1280px left/right split panels + augmentation | Full training run | mAP50: **0.71**, mAP50-95: 0.45 on the *same* station-disjoint held-out split — best L/R teacher; augmentation genuinely helped (+0.05). |

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
