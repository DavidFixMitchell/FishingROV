#!/usr/bin/env python3
"""Run LR-parallel YOLO(1cls) -> crop -> Swin classifier and export VIAME CSV.

Supports two input modes:
- panel: images resolved from dataset.yaml split
- video: sampled frames from a video file (for smoke or batch processing)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


def _canonical_label(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def parse_class_thresholds(spec: str, eval_classes: List[str]) -> Dict[str, float]:
    """Parse class thresholds from 'class=value,class=value' text."""
    out: Dict[str, float] = {}
    if not spec.strip():
        return out

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if "=" not in part:
            raise RuntimeError(f"Invalid class threshold entry '{part}'. Expected class=value.")
        k, v = part.split("=", 1)
        cls = _canonical_label(k)
        try:
            thr = float(v)
        except ValueError as e:
            raise RuntimeError(f"Invalid threshold value for '{k}': {v}") from e
        if thr <= 0:
            raise RuntimeError(f"Threshold for '{k}' must be > 0, got {thr}")
        out[cls] = thr

    allowed = {_canonical_label(c) for c in eval_classes}
    unknown = sorted([k for k in out.keys() if k not in allowed])
    if unknown:
        raise RuntimeError(f"Unknown classes in --class-thresholds: {unknown}")

    return out


def _resolve_panel_images(data_yaml: Path, split: str) -> List[Path]:
    import yaml

    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    root = Path(data.get("path", ""))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    split_dir = root / "images" / split
    if not split_dir.is_dir():
        raise RuntimeError(f"Split directory not found: {split_dir}")

    images: List[Path] = []
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        images.extend(sorted(split_dir.glob(pat)))

    if not images:
        raise RuntimeError(f"No images found in split directory: {split_dir}")

    return sorted(set(images))


def _read_video_frames(
    video_path: Path,
    start_sec: float,
    seconds: float,
    fps_sample: float,
    max_frames: int,
) -> List[Tuple[str, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0:
        src_fps = 25.0

    start_frame = max(0, int(round(start_sec * src_fps)))
    end_frame = int(round((start_sec + seconds) * src_fps)) if seconds > 0 else -1
    if fps_sample <= 0:
        fps_sample = 1.0
    step = max(1, int(round(src_fps / fps_sample)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    frames: List[Tuple[str, np.ndarray]] = []
    frame_idx = start_frame
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if end_frame >= 0 and frame_idx > end_frame:
            break

        if (frame_idx - start_frame) % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image_id = f"{video_path.stem}_f{frame_idx:06d}.jpg"
            frames.append((image_id, rgb))
            if max_frames > 0 and len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    if not frames:
        raise RuntimeError("No frames sampled from video")
    return frames


def load_swin_model(ckpt_path: Path) -> Tuple[torch.nn.Module, int, List[str], torch.device]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" not in ckpt:
        raise RuntimeError(f"Classifier checkpoint missing model metadata: {ckpt_path}")
    if "imgsz" not in ckpt:
        raise RuntimeError(f"Classifier checkpoint missing imgsz metadata: {ckpt_path}")

    model_name = str(ckpt["model"])
    imgsz = int(ckpt["imgsz"])
    class_to_idx = ckpt.get("class_to_idx")
    if not class_to_idx:
        raise RuntimeError("Classifier checkpoint missing class_to_idx")

    classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    model = timm.create_model(model_name, pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"], strict=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, imgsz, classes, device


def square_crop_with_pad(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, fill: int = 114) -> np.ndarray | None:
    h, w = img.shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    side = max(bw, bh)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    sx1 = int(round(cx - side / 2.0))
    sy1 = int(round(cy - side / 2.0))
    sx2 = sx1 + side
    sy2 = sy1 + side

    canvas = np.full((side, side, 3), fill, dtype=np.uint8)

    ix1 = max(0, sx1)
    iy1 = max(0, sy1)
    ix2 = min(w, sx2)
    iy2 = min(h, sy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    dx1 = ix1 - sx1
    dy1 = iy1 - sy1
    dx2 = dx1 + (ix2 - ix1)
    dy2 = dy1 + (iy2 - iy1)
    canvas[dy1:dy2, dx1:dx2] = img[iy1:iy2, ix1:ix2]
    return canvas


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(a_area + b_area - inter)


def nms_xyxy(boxes: Sequence[Tuple[int, int, int, int]], scores: Sequence[float], iou_thr: float) -> List[int]:
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        rem: List[int] = []
        for j in order:
            if bbox_iou(boxes[i], boxes[j]) < iou_thr:
                rem.append(j)
        order = rem
    return keep


def classify_batch(
    model: torch.nn.Module,
    device: torch.device,
    tfm: transforms.Compose,
    crops: List[np.ndarray],
    class_names: List[str],
    class_thresholds: Dict[str, float],
) -> List[Dict[str, Any]]:
    if not crops:
        return []

    imgs = []
    for crop in crops:
        ten = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        imgs.append(tfm(ten))

    batch = torch.stack(imgs, dim=0).to(device, non_blocking=True)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        thr_vec = torch.tensor(
            [float(class_thresholds.get(_canonical_label(c), 1.0)) for c in class_names],
            dtype=probs.dtype,
            device=probs.device,
        )
        norm_scores = probs / thr_vec.unsqueeze(0)
        pred_idx = norm_scores.argmax(dim=1)
        top1_prob = probs.gather(1, pred_idx.unsqueeze(1)).squeeze(1)
        top1_norm = norm_scores.gather(1, pred_idx.unsqueeze(1)).squeeze(1)

    pred_idx_list = pred_idx.detach().cpu().numpy().tolist()
    top1_list = top1_prob.detach().cpu().numpy().tolist()
    top1_norm_list = top1_norm.detach().cpu().numpy().tolist()
    probs_list = probs.detach().cpu().numpy().tolist()
    norm_list = norm_scores.detach().cpu().numpy().tolist()

    out: List[Dict[str, Any]] = []
    for i, cls_idx in enumerate(pred_idx_list):
        out.append(
            {
                "pred_name": class_names[int(cls_idx)],
                "pred_idx": int(cls_idx),
                "pred_conf": float(top1_list[i]),
                "pred_conf_norm": float(top1_norm_list[i]),
                "probs": [float(x) for x in probs_list[i]],
                "norm_scores": [float(x) for x in norm_list[i]],
            }
        )
    return out


def _run_lr_parallel_yolo(
    yolo: YOLO,
    items: List[Tuple[str, np.ndarray]],
    conf: float,
    imgsz: int,
    yolo_batch: int,
    left_xmin: int,
    left_xmax: int,
    right_xmin: int,
    right_xmax: int,
    roi_ymin: int,
    roi_ymax: int,
    merge_iou: float,
) -> List[Tuple[str, np.ndarray, List[Tuple[int, int, int, int]], List[float]]]:
    if not items:
        return []

    # Build crop list: [img0-left, img0-right, img1-left, img1-right, ...]
    crop_sources: List[np.ndarray] = []
    crop_meta: List[Tuple[int, int]] = []  # (item_idx, x_offset)

    for i, (_name, img) in enumerate(items):
        h, w = img.shape[:2]
        y0 = max(0, min(h, roi_ymin))
        y1 = h if roi_ymax <= 0 else max(y0 + 1, min(h, roi_ymax))

        lx0 = max(0, min(w - 1, left_xmin))
        lx1 = max(lx0 + 1, min(w, left_xmax))
        rx0 = max(0, min(w - 1, right_xmin))
        rx1 = max(rx0 + 1, min(w, right_xmax))

        l_crop = img[y0:y1, lx0:lx1]
        r_crop = img[y0:y1, rx0:rx1]

        crop_sources.append(l_crop)
        crop_meta.append((i, lx0))
        crop_sources.append(r_crop)
        crop_meta.append((i, rx0))

    preds_iter = yolo.predict(
        source=crop_sources,
        conf=conf,
        imgsz=imgsz,
        batch=max(1, min(len(crop_sources), yolo_batch * 2)),
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
        stream=True,
    )

    merged_boxes: List[List[Tuple[int, int, int, int]]] = [[] for _ in items]
    merged_scores: List[List[float]] = [[] for _ in items]

    for crop_idx, pred in enumerate(preds_iter):
        item_idx, x_off = crop_meta[crop_idx]
        boxes = pred.boxes
        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy()
        for k in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[k].tolist()
            merged_boxes[item_idx].append((int(x1 + x_off), int(y1 + roi_ymin), int(x2 + x_off), int(y2 + roi_ymin)))
            merged_scores[item_idx].append(float(confs[k]))

    out: List[Tuple[str, np.ndarray, List[Tuple[int, int, int, int]], List[float]]] = []
    for i, (name, img) in enumerate(items):
        boxes_i = merged_boxes[i]
        scores_i = merged_scores[i]

        keep = nms_xyxy(boxes_i, scores_i, merge_iou)
        kept_boxes = [boxes_i[k] for k in keep]
        kept_scores = [scores_i[k] for k in keep]

        # Keep deterministic order by score desc.
        order = np.argsort(-np.array(kept_scores)) if kept_scores else np.array([], dtype=int)
        det_boxes = [kept_boxes[int(k)] for k in order]
        det_scores = [float(kept_scores[int(k)]) for k in order]
        out.append((name, img, det_boxes, det_scores))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="LR cascade inference export: YOLO + Swin -> VIAME CSV")
    parser.add_argument("--yolo-weights", type=str, required=True)
    parser.add_argument("--swin-weights", type=str, required=True)

    parser.add_argument("--input-mode", type=str, default="panel", choices=["panel", "video"])
    parser.add_argument("--panel-data-yaml", type=str, default="")
    parser.add_argument("--panel-split", type=str, default="test")

    parser.add_argument("--video-input", type=str, default="")
    parser.add_argument("--video-start-sec", type=float, default=0.0)
    parser.add_argument("--video-seconds", type=float, default=60.0)
    parser.add_argument("--video-fps-sample", type=float, default=1.0)

    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--yolo-batch", type=int, default=4)
    parser.add_argument("--cls-batch", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=0)

    # LR split/merge settings
    parser.add_argument("--left-xmin", type=int, default=0)
    parser.add_argument("--left-xmax", type=int, default=1080)
    parser.add_argument("--right-xmin", type=int, default=840)
    parser.add_argument("--right-xmax", type=int, default=1920)
    parser.add_argument("--roi-ymin", type=int, default=0)
    parser.add_argument("--roi-ymax", type=int, default=1080)
    parser.add_argument("--merge-iou", type=float, default=0.45)

    parser.add_argument("--drop-not-scallop", action="store_true", default=True)
    parser.add_argument("--no-drop-not-scallop", dest="drop_not_scallop", action="store_false")
    parser.add_argument(
        "--class-thresholds",
        type=str,
        default="king=0.1,queen=0.7,dead=0.14,not_a_scallop=1.0",
        help="Per-class normalization thresholds as class=value pairs. Prediction uses argmax(softmax_prob / class_threshold).",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--out-viame-csv", type=str, required=True)
    parser.add_argument("--out-summary-json", type=str, default="")
    args = parser.parse_args()

    if args.input_mode == "panel":
        if not args.panel_data_yaml:
            raise RuntimeError("--panel-data-yaml is required for input-mode=panel")
        image_paths = _resolve_panel_images(Path(args.panel_data_yaml), args.panel_split)
        if args.max_images > 0:
            image_paths = image_paths[: args.max_images]

        items: List[Tuple[str, np.ndarray]] = []
        for p in image_paths:
            try:
                arr = np.array(Image.open(p).convert("RGB"))
            except Exception:
                continue
            items.append((p.name, arr))
    else:
        if not args.video_input:
            raise RuntimeError("--video-input is required for input-mode=video")
        items = _read_video_frames(
            Path(args.video_input),
            start_sec=args.video_start_sec,
            seconds=args.video_seconds,
            fps_sample=args.video_fps_sample,
            max_frames=args.max_images,
        )

    if not items:
        raise RuntimeError("No input images/frames available for inference")

    yolo = YOLO(args.yolo_weights)
    swin_model, swin_imgsz, swin_classes, device = load_swin_model(Path(args.swin_weights))

    tfm = transforms.Compose(
        [
            transforms.Resize(int(swin_imgsz * 1.15)),
            transforms.CenterCrop(swin_imgsz),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    out_csv = Path(args.out_viame_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    class_names_canon = [_canonical_label(c) for c in swin_classes]
    class_thresholds = parse_class_thresholds(args.class_thresholds, class_names_canon)

    det_total = 0
    det_written = 0
    dropped_not_scallop = 0
    track_id = 1
    images_total = len(items)
    images_processed = 0

    pred_class_counts: Dict[str, int] = {c: 0 for c in class_names_canon}
    written_class_counts: Dict[str, int] = {c: 0 for c in class_names_canon}
    run_start = time.monotonic()

    with out_csv.open("w", encoding="utf-8") as f:
        for s in range(0, len(items), args.yolo_batch):
            chunk = items[s : s + args.yolo_batch]
            lr_out = _run_lr_parallel_yolo(
                yolo,
                chunk,
                conf=args.conf,
                imgsz=args.imgsz,
                yolo_batch=args.yolo_batch,
                left_xmin=args.left_xmin,
                left_xmax=args.left_xmax,
                right_xmin=args.right_xmin,
                right_xmax=args.right_xmax,
                roi_ymin=args.roi_ymin,
                roi_ymax=args.roi_ymax,
                merge_iou=args.merge_iou,
            )

            for image_id, img, det_boxes, det_confs in lr_out:
                images_processed += 1
                if not det_boxes:
                    if not args.quiet and (
                        images_processed == images_total
                        or (args.progress_every > 0 and images_processed % args.progress_every == 0)
                    ):
                        pct = 100.0 * images_processed / max(1, images_total)
                        print(
                            f"[progress] images {images_processed}/{images_total} ({pct:.1f}%) | "
                            f"det_total={det_total} det_written={det_written}",
                            flush=True,
                        )
                    continue

                det_total += len(det_boxes)

                crops: List[np.ndarray] = []
                kept_idx: List[int] = []
                for i, box in enumerate(det_boxes):
                    crop = square_crop_with_pad(img, *box)
                    if crop is None:
                        continue
                    crops.append(crop)
                    kept_idx.append(i)

                for b in range(0, len(crops), args.cls_batch):
                    crop_batch = crops[b : b + args.cls_batch]
                    idx_batch = kept_idx[b : b + args.cls_batch]
                    cls_preds = classify_batch(swin_model, device, tfm, crop_batch, swin_classes, class_thresholds)

                    for j, cls_pred in enumerate(cls_preds):
                        det_i = idx_batch[j]
                        x1, y1, x2, y2 = det_boxes[det_i]
                        det_conf = det_confs[det_i]
                        cls_name = _canonical_label(str(cls_pred["pred_name"]))
                        cls_conf = float(cls_pred["pred_conf"])
                        cls_conf_norm = float(cls_pred.get("pred_conf_norm", cls_conf))
                        if cls_name in pred_class_counts:
                            pred_class_counts[cls_name] += 1

                        if args.drop_not_scallop and cls_name == "not_a_scallop":
                            dropped_not_scallop += 1
                            continue

                        final_conf = max(0.0, min(1.0, det_conf * cls_conf))

                        row = [
                            str(track_id),
                            image_id,
                            "0",
                            str(int(x1)),
                            str(int(y1)),
                            str(int(x2)),
                            str(int(y2)),
                            f"{final_conf:.6f}",
                            "-1",
                            cls_name,
                            f"{cls_conf:.6f}",
                        ]

                        probs = cls_pred.get("probs", [])
                        for ci, cname in enumerate(class_names_canon):
                            if ci < len(probs):
                                row.extend([cname, f"{float(probs[ci]):.6f}"])

                        f.write(",".join(row) + "\n")
                        det_written += 1
                        if cls_name in written_class_counts:
                            written_class_counts[cls_name] += 1
                        track_id += 1

                if not args.quiet and (
                    images_processed == images_total
                    or (args.progress_every > 0 and images_processed % args.progress_every == 0)
                ):
                    pct = 100.0 * images_processed / max(1, images_total)
                    print(
                        f"[progress] images {images_processed}/{images_total} ({pct:.1f}%) | "
                        f"det_total={det_total} det_written={det_written}",
                        flush=True,
                    )

    elapsed_sec = max(0.0, time.monotonic() - run_start)
    avg_fps = images_processed / elapsed_sec if elapsed_sec > 0 else 0.0
    avg_crops_per_frame = det_total / images_processed if images_processed > 0 else 0.0

    summary = {
        "input_mode": args.input_mode,
        "images": len(items),
        "elapsed_sec": elapsed_sec,
        "avg_fps": avg_fps,
        "avg_crops_per_frame": avg_crops_per_frame,
        "det_total": det_total,
        "det_written": det_written,
        "dropped_not_a_scallop": dropped_not_scallop,
        "yolo_weights": args.yolo_weights,
        "swin_weights": args.swin_weights,
        "panel_data_yaml": args.panel_data_yaml if args.input_mode == "panel" else None,
        "panel_split": args.panel_split if args.input_mode == "panel" else None,
        "video_input": args.video_input if args.input_mode == "video" else None,
        "video_start_sec": args.video_start_sec if args.input_mode == "video" else None,
        "video_seconds": args.video_seconds if args.input_mode == "video" else None,
        "video_fps_sample": args.video_fps_sample if args.input_mode == "video" else None,
        "imgsz": args.imgsz,
        "yolo_batch": args.yolo_batch,
        "cls_batch": args.cls_batch,
        "conf": args.conf,
        "class_thresholds": class_thresholds,
        "drop_not_scallop": args.drop_not_scallop,
        "left_xmin": args.left_xmin,
        "left_xmax": args.left_xmax,
        "right_xmin": args.right_xmin,
        "right_xmax": args.right_xmax,
        "roi_ymin": args.roi_ymin,
        "roi_ymax": args.roi_ymax,
        "merge_iou": args.merge_iou,
        "classes": class_names_canon,
        "pred_class_counts": pred_class_counts,
        "written_class_counts": written_class_counts,
        "out_viame_csv": str(out_csv),
    }

    if args.out_summary_json:
        sp = Path(args.out_summary_json)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
