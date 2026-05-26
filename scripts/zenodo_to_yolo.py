"""
zenodo_to_yolo.py

Converts Zenodo 10156830 scallop dataset (VIAME CSV annotations + PNG frames)
into YOLO format for training YOLO11n.

This handles single-frame images (no composite/depth channel).
Images are resized to 640x640 with letterbox padding.

Zenodo VIAME CSV format (space/comma-separated):
    detection_id, video_id, frame, x1, y1, x2, y2, score, target_length, species_label

Output:
    datasets/scallop_zenodo/
        images/train/  images/val/  images/test/
        labels/train/  labels/val/  labels/test/

Usage:
    python zenodo_to_yolo.py \
    --train-dir "/path/to/zenodo/Training files" \
    --test-dir "/path/to/zenodo/Test files" \
    --output-dir /path/to/output/scallop_zenodo \
        --img-size 640 --val-split 0.15
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def find_csv_files(root_dir):
    """Find all CSV annotation files in the Zenodo directory structure."""
    csvs = []
    for path in Path(root_dir).rglob("*.csv"):
        csvs.append(path)
    return sorted(csvs)


def find_image_files(root_dir):
    """Find all image files, indexed by directory for frame matching."""
    images = {}
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
        for path in Path(root_dir).rglob(ext):
            images[path.stem] = path
    return images


def parse_viame_csv(csv_path):
    """
    Parse VIAME CSV annotations.

    VIAME CSV has comment lines starting with # and data lines:
    detection_id, video_or_image_name, frame, x1, y1, x2, y2, score, target_length, species_label
    """
    annotations = defaultdict(list)

    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(",")
            if len(parts) < 10:
                continue

            try:
                det_id = parts[0].strip()
                video_id = parts[1].strip()
                frame = int(parts[2].strip())
                x1 = float(parts[3].strip())
                y1 = float(parts[4].strip())
                x2 = float(parts[5].strip())
                y2 = float(parts[6].strip())
                score = float(parts[7].strip())
                target_len = parts[8].strip()
                species = parts[9].strip()
            except (ValueError, IndexError):
                continue

            # Use video_id as key (matches image filename in Zenodo)
            annotations[video_id].append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "score": score, "species": species, "frame": frame
            })

    return annotations


def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, target_size=640):
    """
    Convert pixel bbox to YOLO normalized format with letterbox awareness.
    Assumes image will be resized to fit within target_size x target_size
    with letterbox padding (maintaining aspect ratio).
    """
    # Compute letterbox scaling
    scale = min(target_size / img_w, target_size / img_h)
    new_w = img_w * scale
    new_h = img_h * scale
    pad_x = (target_size - new_w) / 2.0
    pad_y = (target_size - new_h) / 2.0

    # Scale bbox to letterboxed coordinates
    sx1 = x1 * scale + pad_x
    sy1 = y1 * scale + pad_y
    sx2 = x2 * scale + pad_x
    sy2 = y2 * scale + pad_y

    # Convert to YOLO format (normalized center + wh)
    cx = (sx1 + sx2) / 2.0 / target_size
    cy = (sy1 + sy2) / 2.0 / target_size
    bw = (sx2 - sx1) / target_size
    bh = (sy2 - sy1) / target_size

    # Clip to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))

    if bw < 0.001 or bh < 0.001:
        return None

    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def letterbox_resize(img, target_size=640):
    """Resize image with letterbox padding to target_size x target_size."""
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas


def process_station(csv_path, image_dir, output_images_dir, output_labels_dir,
                    img_size=640, processed_count=0):
    """Process one station's CSV + images into YOLO format."""
    annotations = parse_viame_csv(csv_path)
    images = find_image_files(image_dir)

    count = 0
    for image_key, bboxes in annotations.items():
        # Try to find the matching image
        img_path = images.get(image_key)
        if img_path is None:
            # Try matching by frame number patterns
            for stem, path in images.items():
                if image_key in stem or stem in image_key:
                    img_path = path
                    break
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Generate YOLO labels
        yolo_lines = []
        for box in bboxes:
            line = bbox_to_yolo(box["x1"], box["y1"], box["x2"], box["y2"],
                                w, h, img_size)
            if line:
                yolo_lines.append(line)

        if not yolo_lines:
            continue

        # Save letterboxed image
        out_name = f"{processed_count + count:06d}"
        letterboxed = letterbox_resize(img, img_size)
        cv2.imwrite(str(output_images_dir / f"{out_name}.jpg"), letterboxed,
                     [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save label
        with open(output_labels_dir / f"{out_name}.txt", "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Convert Zenodo scallop data to YOLO format")
    parser.add_argument("--train-dir", required=True, help="Path to extracted Training files")
    parser.add_argument("--test-dir", required=True, help="Path to extracted Test files")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory")
    parser.add_argument("--img-size", type=int, default=640, help="Target image size")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction of training data for validation")
    args = parser.parse_args()

    output = Path(args.output_dir)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Process training data
    print("=== Processing training data ===")
    train_csvs = find_csv_files(args.train_dir)
    print(f"Found {len(train_csvs)} CSV files in training set")

    # First pass: collect all training images + labels into a temp area
    tmp_train_imgs = output / "images" / "_tmp_train"
    tmp_train_lbls = output / "labels" / "_tmp_train"
    tmp_train_imgs.mkdir(parents=True, exist_ok=True)
    tmp_train_lbls.mkdir(parents=True, exist_ok=True)

    total = 0
    for csv_path in train_csvs:
        # Image dir is typically alongside the CSV
        image_dir = csv_path.parent
        count = process_station(csv_path, image_dir, tmp_train_imgs, tmp_train_lbls,
                                args.img_size, total)
        print(f"  {csv_path.name}: {count} images")
        total += count

    print(f"Total training images: {total}")

    # Split into train/val
    all_files = sorted(tmp_train_imgs.glob("*.jpg"))
    np.random.seed(42)
    indices = np.random.permutation(len(all_files))
    val_count = int(len(all_files) * args.val_split)

    val_indices = set(indices[:val_count])
    for i, img_path in enumerate(all_files):
        split = "val" if i in val_indices else "train"
        label_path = tmp_train_lbls / f"{img_path.stem}.txt"

        shutil.move(str(img_path), str(output / "images" / split / img_path.name))
        if label_path.exists():
            shutil.move(str(label_path), str(output / "labels" / split / label_path.name))

    # Clean up temp dirs
    shutil.rmtree(tmp_train_imgs, ignore_errors=True)
    shutil.rmtree(tmp_train_lbls, ignore_errors=True)

    print(f"Train: {total - val_count} images, Val: {val_count} images")

    # Process test data
    print("\n=== Processing test data ===")
    test_csvs = find_csv_files(args.test_dir)
    print(f"Found {len(test_csvs)} CSV files in test set")

    test_total = 0
    for csv_path in test_csvs:
        image_dir = csv_path.parent
        count = process_station(csv_path, image_dir,
                                output / "images" / "test",
                                output / "labels" / "test",
                                args.img_size, test_total)
        print(f"  {csv_path.name}: {count} images")
        test_total += count

    print(f"Test: {test_total} images")
    print(f"\n=== Done! Dataset at {output} ===")
    print(f"  Train: {total - val_count}  Val: {val_count}  Test: {test_total}")


if __name__ == "__main__":
    main()
