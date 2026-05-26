# Hugging Face Artifacts

FishingROV uses Hugging Face primarily as an artifact host for derivative
datasets and model weights. At this stage, the most useful public links are the
dataset artifacts that let other people inspect or reproduce parts of the
training pipeline without cloning large files into Git.

These artifacts are early-stage research outputs. Treat them as experimental
supporting material, not polished benchmark releases.

## Recommended Public Links

### Datasets

- `FishingROV/scallop_lr_teacher_640`
  - Link: https://huggingface.co/datasets/FishingROV/scallop_lr_teacher_640
  - Purpose: left/right panel derivative of the public Zenodo scallop dataset.
  - Usefulness: good candidate for external validation because it includes the
    derived training structure directly, rather than only a compressed archive.

- `FishingROV/scallop_yolov5s_mosaic_640`
  - Link: https://huggingface.co/datasets/FishingROV/scallop_yolov5s_mosaic_640
  - Purpose: 640px one-class tiled/mosaic derivative for edge-model training.
  - Usefulness: useful for reproducing the mosaic-data idea, though the current
    upload is packaged as a `.tar.gz` archive rather than an expanded dataset.

- `FishingROV/scallop_yolov5s_mosaic_640_4class`
  - Link: https://huggingface.co/datasets/FishingROV/scallop_yolov5s_mosaic_640_4class
  - Purpose: four-class tiled/mosaic derivative used for taxonomic/condition
    experiments.
  - Usefulness: useful for validation of the four-class training setup, again as
    an archive-style release.

### Experimental Support Data

- `FishingROV/scallop_mosaic_640_quantization_sample`
  - Link: https://huggingface.co/datasets/FishingROV/scallop_mosaic_640_quantization_sample
  - Purpose: calibration/quantization sample used for RKNN export work.
  - Public-link recommendation: optional. It is real and public, but it is more
    of a build-support artifact than a headline validation artifact.

## Model Artifacts

- `FishingROV/yolov5s_scallop_mosaic_640`
  - Link: https://huggingface.co/FishingROV/yolov5s_scallop_mosaic_640
  - Status: publicly visible and has a basic model card.
  - Recommendation: acceptable to link as an experimental weight release, but it
    should be described as preliminary and its reported metrics should be checked
    against the current project summary before it becomes a front-page proof
    point.

- `FishingROV/scallop_yolov5s_640_model`
  - Link: https://huggingface.co/FishingROV/scallop_yolov5s_640_model
  - Status: publicly visible but not yet curated as a public reference artifact.
  - Current issues: no useful model card, mixed file formats in one place,
    unclear canonical file, and missing public-facing explanation.
  - Recommendation: do not feature this from the GitHub README yet. Either clean
    it into a versioned model repo or leave it as an internal convenience host.

## Private / Internal Artifacts

Some Hugging Face repositories are being used as private handoff or storage
points between the server, travel laptop, and Aura deployment workflow.

Those should remain private until they have:

- a clean model or dataset card;
- clear provenance and license text;
- a stable intended file to download;
- no operational-only runtime notes or private deployment assumptions.

## Current Recommendation

If the goal is to let someone validate the work at this early stage, link the
public derivative datasets first. They show real transformation work and are the
best starting point for reproduction.

Link model weights more selectively:

- okay to mention the public mosaic model as experimental;
- avoid promoting mixed or passthrough storage repos until they are cleaned up;
- keep private HF repos out of the public GitHub repo until they have been
  turned into intentional public artifacts.
