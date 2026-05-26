#!/usr/bin/env python3
"""Generate a letterboxed rectangular YOLO dataset for a scout model.

This keeps the original image content and aspect ratio, then remaps YOLO labels
into the fixed output canvas. It is intended for a fast scout pass such as
512x288 on edge hardware.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def letterbox_image(img, target_w, target_h):
    src_h, src_w = img.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    pad_x = (target_w - dst_w) // 2
    pad_y = (target_h - dst_h) // 2
    canvas[pad_y:pad_y + dst_h, pad_x:pad_x + dst_w] = resized
    return canvas, scale, pad_x, pad_y


def remap_labels(label_path, src_w, src_h, target_w, target_h, scale, pad_x, pad_y):
    rows = []
    if not label_path.exists():
        return rows

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            continue
        class_id = parts[0]
        xc, yc, box_w, box_h = map(float, parts[1:])
        x1 = (xc - box_w / 2.0) * src_w
        y1 = (yc - box_h / 2.0) * src_h
        x2 = (xc + box_w / 2.0) * src_w
        y2 = (yc + box_h / 2.0) * src_h

        x1 = x1 * scale + pad_x
        y1 = y1 * scale + pad_y
        x2 = x2 * scale + pad_x
        y2 = y2 * scale + pad_y

        x1 = min(max(x1, 0.0), target_w)
        y1 = min(max(y1, 0.0), target_h)
        x2 = min(max(x2, 0.0), target_w)
        y2 = min(max(y2, 0.0), target_h)

        if x2 <= x1 or y2 <= y1:
            continue

        new_xc = ((x1 + x2) * 0.5) / target_w
        new_yc = ((y1 + y2) * 0.5) / target_h
        new_w = (x2 - x1) / target_w
        new_h = (y2 - y1) / target_h
        rows.append(f"{class_id} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}")
    return rows


def copy_split(source_root, output_root, split, target_w, target_h):
    src_images = source_root / "images" / split
    src_labels = source_root / "labels" / split
    dst_images = output_root / "images" / split
    dst_labels = output_root / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in src_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not image_paths:
        raise RuntimeError(f"No images found in {src_images}")

    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Failed to read image {image_path}")
        src_h, src_w = img.shape[:2]
        boxed, scale, pad_x, pad_y = letterbox_image(img, target_w, target_h)
        label_path = src_labels / f"{image_path.stem}.txt"
        rows = remap_labels(label_path, src_w, src_h, target_w, target_h, scale, pad_x, pad_y)

        out_image = dst_images / f"{image_path.stem}.jpg"
        out_label = dst_labels / f"{image_path.stem}.txt"
        if not cv2.imwrite(str(out_image), boxed):
            raise RuntimeError(f"Failed to write image {out_image}")
        out_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def write_dataset_yaml(source_root, output_root, target_w, target_h):
    src_yaml = source_root / "dataset.yaml"
    if src_yaml.exists():
        src_text = src_yaml.read_text(encoding="utf-8")
        names_line = next((line for line in src_text.splitlines() if line.strip().startswith("names:")), "names: ['scallop']")
        nc_line = next((line for line in src_text.splitlines() if line.strip().startswith("nc:")), None)
    else:
        names_line = "names: ['scallop']"
        nc_line = "nc: 1"

    lines = [
        f"path: {output_root.as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if nc_line:
        lines.append(nc_line)
    lines.append(names_line)
    lines.append(f"# scout_input: {target_w}x{target_h}")
    (output_root / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate a rectangular scout YOLO dataset")
    parser.add_argument("--source", required=True, help="Source YOLO dataset root with images/train and labels/train")
    parser.add_argument("--output", required=True, help="Output dataset root")
    parser.add_argument("--width", type=int, required=True, help="Target scout width, for example 512")
    parser.add_argument("--height", type=int, required=True, help="Target scout height, for example 320")
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")

    source_root = Path(args.source)
    output_root = Path(args.output)
    if not source_root.exists():
        raise RuntimeError(f"Source dataset not found: {source_root}")

    for split in ("train", "val"):
        copy_split(source_root, output_root, split, args.width, args.height)
    write_dataset_yaml(source_root, output_root, args.width, args.height)

    readme = output_root / "README.md"
    readme.write_text(
        "\n".join([
            f"# Rectangular Scout Dataset",
            "",
            f"Derived from: {source_root.as_posix()}",
            f"Target input: {args.width}x{args.height}",
            "Transform: aspect-preserving letterbox with YOLO labels remapped into the new canvas.",
        ]) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()