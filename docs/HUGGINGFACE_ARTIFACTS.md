# Hugging Face Artifacts

FishingROV uses Hugging Face primarily as an artifact host for derivative
datasets and model weights. At this stage, the most useful public links are the
dataset artifacts that let other people inspect or reproduce parts of the
training pipeline without cloning large files into Git.

These artifacts are early-stage research outputs. Treat them as experimental
supporting material, not polished benchmark releases.

## Current Public Inventory (as of 2026-06-01)

### Models

#### FishingROV/scallop_yolo26x_lr_1280_aug

- Link: https://huggingface.co/FishingROV/scallop_yolo26x_lr_1280_aug
- Role: 3090-side teacher detector (best honest L/R teacher; mAP50 0.705 held-out)
- File count: 12
- Files:
  - .gitattributes
  - README.md
  - best.pt
  - last.pt
  - args.yaml
  - results.csv
  - results.png
  - BoxPR_curve.png
  - BoxF1_curve.png
  - confusion_matrix_normalized.png
  - val_batch0_pred.jpg
  - val_batch0_labels.jpg

#### FishingROV/yolov5s_scallop_mosaic_640

- Link: https://huggingface.co/FishingROV/yolov5s_scallop_mosaic_640
- File count: 3
- Files:
  - .gitattributes
  - README.md
  - yolov5s_scallop_mosaic_640_best.pt

#### FishingROV/scallop_yolov5s_640_model

- Link: https://huggingface.co/FishingROV/scallop_yolov5s_640_model
- File count: 10
- Files:
  - .gitattributes
  - best.onnx
  - best.pt
  - best.rknn
  - best.torchscript
  - best_rv1126b.rknn
  - best_rv1126b_mmse.rknn
  - best_split.onnx
  - best_split_rv1126b.rknn
  - calib.txt

#### FishingROV/scallop_yolo26x_lr_1280_v2

- Link: https://huggingface.co/FishingROV/scallop_yolo26x_lr_1280_v2
- File count: 6
- Files:
  - .gitattributes
  - README.md
  - best.pt
  - class_eval_best.json
  - public_FishingROV_model_card_example.md
  - results.csv

#### FishingROV/classifier_swinv2b_256

- Link: https://huggingface.co/FishingROV/classifier_swinv2b_256
- File count: 7
- Files:
  - .gitattributes
  - README.md
  - best.pt
  - class_eval_best.json
  - classes.json
  - history.json
  - public_FishingROV_model_card_example.md

### Datasets

#### FishingROV/scallop_mosaic_640_quantization_sample

- Link: https://huggingface.co/datasets/FishingROV/scallop_mosaic_640_quantization_sample
- File count: 2002
- Split layout: train + val directories present
- Top-level files:
  - .gitattributes
  - README.md

#### FishingROV/scallop_yolov5s_mosaic_640

- Link: https://huggingface.co/datasets/FishingROV/scallop_yolov5s_mosaic_640
- File count: 3
- Split layout: archive-style release (no expanded train/val directories)
- Files:
  - .gitattributes
  - README.md
  - scallop_yolov5s_mosaic_640.tar.gz

#### FishingROV/scallop_yolov5s_mosaic_640_4class

- Link: https://huggingface.co/datasets/FishingROV/scallop_yolov5s_mosaic_640_4class
- File count: 3
- Split layout: archive-style release (no expanded train/val directories)
- Files:
  - .gitattributes
  - README.md
  - scallop_yolov5s_mosaic_640_4class.tar.gz

#### FishingROV/scallop_lr_teacher_640

- Link: https://huggingface.co/datasets/FishingROV/scallop_lr_teacher_640
- File count: 11482
- Split layout: images/train + images/val directories present
- Top-level files:
  - .gitattributes
  - README.md
  - dataset.yaml
  - pipeline_meta.json

## Notes

- Inventory is pulled from the live Hugging Face API for author FishingROV.
- Large expanded datasets are summarized by structure and top-level files for
  readability.
