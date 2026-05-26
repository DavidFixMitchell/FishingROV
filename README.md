# FishingROV
Ultimate project goal is to see selective king scallop harvesting become an economic reality, offering an alternative to scallop dredge gear.

Open-source experiments for detecting scallops in underwater video using public
survey imagery, YOLO-style object detection, and low-cost edge deployment.

The immediate goal is to get more data, particularly images containing king scallops. Pseudo-label and retrain detectors.

The ROV edge-deployment work is for pipeline validation, experimentation, and training-data collection, while building the platform for catch-mechanism testing and refining the real-time detection pipeline.

## Current Focus

- Convert public VIAME/DIVE scallop annotations into YOLO training labels.
- Train and evaluate scallop detectors on the public St Andrews / NatureScot
  Zenodo dataset.
- Explore a two-stage edge pipeline: fast rectangular scout detector, then a
  higher-resolution crop/mosaic detector for confirmation.
- Convert suitable exported models to Rockchip RKNN for low-cost ROV computers.
- Build a practical pseudo-labelling workflow for additional unlabelled survey
  video, with human review of uncertain detections.
- Deploy the ROV for data gathering, model refinement, and full 4K image processing.

## Public Dataset

Primary public data source:

- Harlow, L., Ovchinnikova, K., & James, M. (2023). Data for Scallop
  (Pecten maximus) Identification in Natural Marine Habitats: A NetHarn Model
  Approach. Zenodo. https://doi.org/10.5281/zenodo.10156830

Related paper:

- Harlow, L., Ovchinnikova, K., & James, M. (2025). Neural network-based
  identification for scallops (Pecten maximus) in natural marine habitats.
  PLOS ONE. https://doi.org/10.1371/journal.pone.0327824

The dataset and paper are CC-BY-4.0. This repository does not redistribute raw
datasets, trained weights, or large model artifacts.

## Repository Contents

- `docs/PROJECT_PROGRESS.md` - current work completed: training server,
  dataset translation/augmentation, preliminary model results, Aura deployment,
  and Hugging Face artifact posture.
- `docs/MODEL_INDEX.md` - sanitized public summary of model families,
  preliminary results, deployment status, and next architecture.
- `docs/HUGGINGFACE_ARTIFACTS.md` - audit of public artifact-hosted datasets and
  weights, including which ones are worth linking for early validation.
- `scripts/zenodo_to_yolo.py` - convert Zenodo VIAME CSV annotations and frames
  into a YOLO-style dataset.
- `scripts/viame_to_yolo_composite.py` - convert VIAME annotations into labels
  for a 640x640 RGB/depth composite frame.
- `scripts/make_composite_frames.py` - build RGB/depth composite frames.
- `scripts/generate_rect_scout_dataset.py` - remap a YOLO dataset into a fixed
  rectangular scout input such as 512x288.
- `scripts/convert_onnx_to_rknn.py` - convert ONNX models to RKNN for Rockchip
  targets when RKNN Toolkit2 is available.
- `docs/data/DATA_ACCESS.md` - public data access, attribution, and artifact
  sharing guidance.

## What Is Not Included

- Raw survey video or images.
- Training datasets or generated labels.
- Model weights (`.pt`, `.onnx`, `.rknn`).
- Private server paths, deployment keys, credentials, or local machine setup.
- Operational dashboards and device-specific runtime files.

## Status

Early prototype with working dataset conversion, training, and Aura deployment
experiments. Preliminary progress is summarized in `docs/PROJECT_PROGRESS.md`
and `docs/MODEL_INDEX.md`. Use the code as a starting point, not as a finished
stock-assessment product.
