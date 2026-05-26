"""
make_composite_frames.py

Takes left camera frames + corresponding depth/disparity map frames and creates
640x640 composite images for YOLOv5 training/inference.

The RGB image keeps its native aspect ratio (no distortion), filling the top
of the 640x640 frame. The disparity map is squished to fill the remaining
space at the bottom. This exploits YOLOv5's letterbox padding — pixels that
would otherwise be wasted gray padding now carry depth information for free.

For 16:9 source (1920x1080 / IMX415):
  Top  (640x360): left camera image at native aspect
  Bottom (640x280): disparity map squished to fill remainder

Usage:
    python make_composite_frames.py \
        --left-dir frames/left/ \
        --depth-dir frames/depth/ \
        --output-dir frames/composite/ \
        --colorize-depth
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def make_composite(left_img, depth_img, size=640, colorize_depth=False):
    """Create a 640x640 composite from left camera and disparity images.

    The left camera image is scaled to 640 wide at its native aspect ratio
    (e.g. 16:9 → 640x360). The disparity map is squished into the remaining
    vertical space (e.g. 640x280). This fills the YOLOv5 letterbox padding
    with useful depth data instead of wasting it on gray pixels.
    """
    src_h, src_w = left_img.shape[:2]
    # Scale to 640 wide, preserving aspect ratio
    rgb_h = int(round(src_h * size / src_w))
    depth_h = size - rgb_h  # remaining space for disparity

    # Resize left camera at native aspect (no distortion)
    left_resized = cv2.resize(left_img, (size, rgb_h), interpolation=cv2.INTER_LINEAR)

    # Process depth / disparity
    if colorize_depth:
        if len(depth_img.shape) == 2:
            depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        else:
            depth_colored = depth_img
        depth_resized = cv2.resize(depth_colored, (size, depth_h), interpolation=cv2.INTER_LINEAR)
    else:
        if len(depth_img.shape) == 2:
            depth_3ch = cv2.cvtColor(depth_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            depth_3ch = depth_img
        depth_resized = cv2.resize(depth_3ch, (size, depth_h), interpolation=cv2.INTER_LINEAR)

    # Stack vertically: RGB on top, squished disparity on bottom
    composite = np.vstack([left_resized, depth_resized])
    return composite


def main():
    parser = argparse.ArgumentParser(description="Create composite frames for YOLOv5")
    parser.add_argument("--left-dir", required=True, help="Directory of left camera frames")
    parser.add_argument("--depth-dir", required=True, help="Directory of depth map frames")
    parser.add_argument("--output-dir", required=True, help="Output directory for composites")
    parser.add_argument("--colorize-depth", action="store_true", help="Apply JET colormap to depth")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    left_files = sorted(Path(args.left_dir).glob("*.png")) + sorted(Path(args.left_dir).glob("*.jpg"))
    depth_files = sorted(Path(args.depth_dir).glob("*.png")) + sorted(Path(args.depth_dir).glob("*.jpg"))

    if len(left_files) != len(depth_files):
        print(f"WARNING: {len(left_files)} left frames vs {len(depth_files)} depth frames")

    count = min(len(left_files), len(depth_files))
    for i in range(count):
        left_img = cv2.imread(str(left_files[i]))
        depth_img = cv2.imread(str(depth_files[i]), cv2.IMREAD_ANYDEPTH)
        if depth_img is None:
            depth_img = cv2.imread(str(depth_files[i]))

        composite = make_composite(left_img, depth_img, colorize_depth=args.colorize_depth)

        out_path = Path(args.output_dir) / f"frame_{i:06d}.jpg"
        cv2.imwrite(str(out_path), composite)

    print(f"Created {count} composite frames in {args.output_dir}")


if __name__ == "__main__":
    main()
