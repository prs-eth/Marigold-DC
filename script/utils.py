import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def validate_rgb_sparse_structure(input_dir: str) -> Tuple[bool, str]:
    """
    Validate that the input directory contains RGB images and corresponding sparse depth files.
    Each RGB image should have a corresponding sparse depth file.

    Args:
        input_dir: Path to the input directory with rgb/ and sparse/ subdirectories

    Returns:
        Tuple of (success: bool, error_message: str). If successful, error_message is empty string.
    """
    input_path = Path(input_dir)
    rgb_dir = input_path / "rgb"
    sparse_dir = input_path / "sparse"

    if not rgb_dir.exists():
        return False, f"RGB directory not found: {rgb_dir}"
    if not sparse_dir.exists():
        return False, f"Sparse directory not found: {sparse_dir}"

    # Get all files from each directory
    rgb_files = sorted([f.stem for f in rgb_dir.glob("*.png")] + [f.stem for f in rgb_dir.glob("*.jpg")] + [f.stem for f in rgb_dir.glob("*.jpeg")])
    sparse_files = sorted([f.stem for f in sparse_dir.glob("*.npy")])

    if not rgb_files:
        return False, f"No RGB images found in {rgb_dir}"
    if not sparse_files:
        return False, f"No sparse depth files found in {sparse_dir}"

    # Check if RGB and sparse directories have the same set of filenames
    if set(rgb_files) != set(sparse_files):
        missing_rgb = set(sparse_files) - set(rgb_files)
        missing_sparse = set(rgb_files) - set(sparse_files)

        error_msg = "Filename mismatch between RGB and sparse directories:\n"
        if missing_rgb:
            error_msg += f"  Missing RGB files: {missing_rgb}\n"
        if missing_sparse:
            error_msg += f"  Missing sparse files: {missing_sparse}\n"

        return False, error_msg

    logging.info(f"RGB and sparse validation successful. Found {len(rgb_files)} files.")
    return True, ""


def validate_prediction_gt_structure(prediction_dir: str, gt_dir: str) -> Tuple[bool, str]:
    """
    Validate that the prediction and GT directories contain matching depth files.
    Each prediction file should have a corresponding GT depth file.

    Args:
        prediction_dir: Path to the prediction directory
        gt_dir: Path to the GT directory

    Returns:
        Tuple of (success: bool, error_message: str). If successful, error_message is empty string.
    """
    prediction_path = Path(prediction_dir)
    gt_path = Path(gt_dir)

    if not prediction_path.exists():
        return False, f"Prediction directory not found: {prediction_path}"
    if not gt_path.exists():
        return False, f"GT directory not found: {gt_path}"

    prediction_files = sorted([f.stem for f in prediction_path.glob("*.npy")])
    gt_files = sorted([f.stem for f in gt_path.glob("*.npy")])

    if not prediction_files:
        return False, f"No prediction files found in {prediction_path}"
    if not gt_files:
        return False, f"No GT depth files found in {gt_path}"

    # Check if prediction and GT directories have the same set of filenames
    if set(prediction_files) != set(gt_files):
        missing_prediction = set(gt_files) - set(prediction_files)
        missing_gt = set(prediction_files) - set(gt_files)

        error_msg = "Filename mismatch between prediction and GT directories:\n"
        if missing_prediction:
            error_msg += f"  Missing prediction files: {missing_prediction}\n"
        if missing_gt:
            error_msg += f"  Missing GT files: {missing_gt}\n"

        return False, error_msg

    logging.info(f"Prediction and GT validation successful. Found {len(prediction_files)} files.")
    return True, ""
