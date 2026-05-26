"""
viame_to_yolo_composite.py

Converts VIAME CSV annotations from standard video frames into YOLO-format
labels for the 640x640 composite frame.

The composite uses the camera's native aspect ratio for RGB (e.g. 16:9 = 640x360)
and squishes the disparity map into the remaining space (640x280). This exploits
YOLOv5's letterbox padding — pixels that would be wasted on gray now carry depth.

For 1920x1080 source:
  Top  (640x360): left camera image at native 16:9 aspect
  Bottom (640x280): disparity map squished to fill remainder

Usage:
    python viame_to_yolo_composite.py \
        --viame-csv annotations.csv \
        --video-width 1920 --video-height 1080 \
        --output-dir labels/

VIAME CSV format (columns):
    detection_id, video_id, frame, x1, y1, x2, y2, score, target_length, species_label

Output YOLO format (per frame .txt):
    class x_center y_center width height  (all normalized 0-1 relative to 640x640)
"""

import argparse
import csv
import os
from pathlib import Path


def viame_bbox_to_composite_yolo(x1, y1, x2, y2, src_w, src_h, composite_size=640):
    """
    Convert a VIAME bounding box (in original video pixel coords) to two YOLO
    bounding boxes in the 640x640 composite frame.

    The composite uses native aspect ratio for the RGB portion:
      - RGB: 640 x rgb_h  (e.g. 360 for 16:9 source)
      - Depth: 640 x depth_h  (e.g. 280 = 640 - 360)

    Returns list of YOLO lines: "class x_center y_center width height"
    """
    # Compute split based on source aspect ratio
    rgb_h = int(round(src_h * composite_size / src_w))  # e.g. 360 for 16:9
    depth_h = composite_size - rgb_h  # e.g. 280

    # Scale from source video to RGB region (top portion)
    scale_x = composite_size / src_w
    scale_y = rgb_h / src_h  # same as scale_x for native aspect

    # Scaled bbox in composite pixel coords (top / RGB region)
    cx = ((x1 + x2) / 2.0) * scale_x
    cy_rgb = ((y1 + y2) / 2.0) * scale_y
    bw = (x2 - x1) * scale_x
    bh_rgb = (y2 - y1) * scale_y

    # Normalize to 0-1 for YOLO (relative to 640x640)
    norm_cx = cx / composite_size
    norm_w = bw / composite_size

    # Top (RGB) label — y range is 0 to rgb_h/640 (e.g. 0 to 0.5625)
    norm_cy_rgb = cy_rgb / composite_size
    norm_h_rgb = bh_rgb / composite_size

    # Bottom (depth) label — squished vertically by depth_h/rgb_h ratio
    # The bbox y-coordinate maps into the depth region (starts at rgb_h)
    squish_ratio = depth_h / rgb_h  # e.g. 280/360 ≈ 0.778
    cy_depth = rgb_h + (cy_rgb * squish_ratio)  # offset into depth zone
    bh_depth = bh_rgb * squish_ratio  # squished height

    norm_cy_depth = cy_depth / composite_size
    norm_h_depth = bh_depth / composite_size

    labels = []
    labels.append(f"0 {norm_cx:.6f} {norm_cy_rgb:.6f} {norm_w:.6f} {norm_h_rgb:.6f}")
    labels.append(f"0 {norm_cx:.6f} {norm_cy_depth:.6f} {norm_w:.6f} {norm_h_depth:.6f}")
    return labels


def parse_viame_csv(csv_path):
    """Parse VIAME CSV and group bounding boxes by frame number."""
    frames = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 10:
                continue
            frame_num = int(row[2])
            x1 = float(row[3])
            y1 = float(row[4])
            x2 = float(row[5])
            y2 = float(row[6])
            label = row[9].strip()

            if frame_num not in frames:
                frames[frame_num] = []
            frames[frame_num].append((x1, y1, x2, y2, label))
    return frames


def main():
    parser = argparse.ArgumentParser(description="Convert VIAME annotations to YOLO composite format")
    parser.add_argument("--viame-csv", required=True, help="Path to VIAME CSV annotations")
    parser.add_argument("--video-width", type=int, default=1920, help="Source video width")
    parser.add_argument("--video-height", type=int, default=1080, help="Source video height")
    parser.add_argument("--output-dir", required=True, help="Output directory for YOLO .txt labels")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    frames = parse_viame_csv(args.viame_csv)

    for frame_num, detections in frames.items():
        label_lines = []
        for x1, y1, x2, y2, label in detections:
            lines = viame_bbox_to_composite_yolo(
                x1, y1, x2, y2, args.video_width, args.video_height
            )
            label_lines.extend(lines)

        out_path = Path(args.output_dir) / f"frame_{frame_num:06d}.txt"
        with open(out_path, "w") as f:
            f.write("\n".join(label_lines) + "\n")

    print(f"Converted {len(frames)} frames to YOLO format in {args.output_dir}")


if __name__ == "__main__":
    main()
