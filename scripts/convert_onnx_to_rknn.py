#!/usr/bin/env python3
"""Convert ONNX model to RKNN for RV1126 (Aura board).

Requirements:
    - Python 3.8-3.10 (rknn-toolkit2 requirement)
    - pip install rknn-toolkit2  (from Rockchip GitHub releases)
      https://github.com/airockchip/rknn-toolkit2/releases

Usage:
    python scripts/convert_onnx_to_rknn.py --onnx exports/yolov5s_640_stock/yolov5su.onnx
    python scripts/convert_onnx_to_rknn.py --onnx exports/yolov5s_640_best/best.onnx --quantize

Notes:
    - RV1126 NPU supports INT8 quantization (--quantize) for best throughput
    - Without --quantize, FP16 is used (slower but no calibration needed)
    - For INT8, provide --dataset with a text file listing calibration image paths
      (100-200 representative images from your dataset, one path per line)
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to RKNN for RV1126")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--output", default=None, help="Output .rknn path")
    parser.add_argument("--quantize", action="store_true",
                        help="Enable INT8 quantization (needs --dataset)")
    parser.add_argument("--dataset", default=None,
                        help="Text file with calibration image paths (for INT8)")
    parser.add_argument("--target", default="rv1126",
                        help="Target platform (default: rv1126)")
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"Error: ONNX not found: {onnx_path}")
        sys.exit(1)

    if args.quantize and not args.dataset:
        print("Error: --quantize requires --dataset (calibration images list)")
        print("Generate with: find /path/to/images -name '*.jpg' | head -200 > calib_images.txt")
        sys.exit(1)

    try:
        from rknn.api import RKNN
    except ImportError:
        print("Error: rknn-toolkit2 not installed.")
        print("Requires Python 3.8-3.10. Install from:")
        print("  https://github.com/airockchip/rknn-toolkit2/releases")
        sys.exit(1)

    output = Path(args.output) if args.output else onnx_path.with_suffix(".rknn")

    rknn = RKNN(verbose=True)

    # Config
    print(f"Target: {args.target}")
    print(f"Quantize: {'INT8' if args.quantize else 'FP16'}")

    rknn.config(
        mean_values=[[0, 0, 0]],         # Ultralytics normalizes to 0-1
        std_values=[[255, 255, 255]],     # so mean=0, std=255 maps [0,255]->[0,1]
        target_platform=args.target,
        quantized_dtype="asymmetric_quantized-8" if args.quantize else None,
        optimization_level=3,
    )

    # Load ONNX
    print(f"Loading ONNX: {onnx_path}")
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        print(f"Error loading ONNX: {ret}")
        sys.exit(1)

    # Build
    print("Building RKNN model...")
    ret = rknn.build(
        do_quantization=args.quantize,
        dataset=args.dataset,
    )
    if ret != 0:
        print(f"Error building RKNN: {ret}")
        sys.exit(1)

    # Export
    ret = rknn.export_rknn(str(output))
    if ret != 0:
        print(f"Error exporting: {ret}")
        sys.exit(1)

    import os
    size_mb = os.path.getsize(output) / 1e6
    print(f"\nRKNN exported: {output} ({size_mb:.1f} MB)")
    print(f"Deploy to Aura board with rknn-toolkit-lite2 or C API")

    rknn.release()


if __name__ == "__main__":
    main()
