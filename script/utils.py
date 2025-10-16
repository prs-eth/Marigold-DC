import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numba import njit


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


def deterministic_select_k_true(valid_mask: np.ndarray, k: Optional[int] = None, seed: int = 2024) -> np.ndarray:
    """
    Select exactly k True values from a boolean mask. Used to sample the sparse depth values from the GT depth map.

    Args:
        valid_mask (np.ndarray): A boolean 2D numpy array.
        k (Optional[int]): The number of True values to select. If None, return the original mask.
        seed (int): Seed for the random number generator.

    Returns:
        np.ndarray: A boolean 2D numpy array with exactly k True values, subset of the input mask.
    """
    num_true = np.sum(valid_mask)
    true_indices = np.argwhere(valid_mask)

    # If k is None, return the original mask
    if k is None:
        logging.warning("k is None, returning the original mask")
        return valid_mask

    # If k is 0 or the mask has no True values, return an empty mask
    if k == 0 or num_true == 0:
        logging.warning("k is 0 or the mask has no True values, returning an empty mask")
        return np.zeros_like(valid_mask, dtype=bool)

    # Ensure k is not greater than the number of True values in the mask
    if k > num_true:
        logging.warning(f"k is greater than the number of True values in the mask, setting k to {num_true}")
        k = num_true

    # Pick the first k indices after shuffling
    np.random.seed(seed)
    np.random.shuffle(true_indices)
    selected_indices = true_indices[:k]

    selected_arr = np.zeros_like(valid_mask, dtype=bool)
    selected_arr[tuple(zip(*selected_indices))] = True
    return selected_arr


# Code for filtering sourced from https://github.com/bartn8/vppdc/
@njit
def filter(dmap, conf_map, th):
    """
    Drop points from a disparity map based on a confidence map.

    Parameters
    ----------
    dmap: HxW np.ndarray
        Disparity map to modify: there is side-effect.
    conf_map: HxW np.ndarray
        Confidence map to use for filtering (1 if point is filtered).
    th: float
        Threshold for filtering

    Returns
    -------
    filtered_i: int
        Number of points filtered
    """
    h, w = dmap.shape[:2]
    filtered_i = 0
    for y in range(h):
        for x in range(w):
            if dmap[y, x] > 0:
                if conf_map[y, x] > th:
                    dmap[y, x] = 0
                    filtered_i += 1
    return filtered_i


@njit
def conti_conf_depth(delta_map, th=3):
    """
    Return a confidence map based on Conti's method (https://arxiv.org/abs/2210.03118).
    Points in a window that are far from foreground are rejected.
    Parameters
    ----------
    dmap: HxW np.ndarray
        Depth map used to extract confidence map.
    n: int
        Window size (3,5,7,...)
    th: float
        Threshold for absolute difference
    Returns
    -------
    conf_rst: HxW np.ndarray
        Binary confidence map (1 for rejected points)
    """
    h, w = delta_map.shape[:2]

    # Confidence map between 0 and 1 (binary)
    conf_map = np.zeros(delta_map.shape, dtype=np.uint8)

    # Conti's filtering method
    for y in range(h):
        for x in range(w):
            # Absolute thresholding
            if delta_map[y, x] > th:
                conf_map[y, x] = 1

    return conf_map


@njit
def delta_depth(dmap, nx=7, ny=3):
    """
    Return a confidence map based on Conti's method (https://arxiv.org/abs/2210.03118).
    Points in a window that are far from foreground are rejected.
    Parameters
    ----------
    dmap: HxW np.ndarray
        Depth map used to extract confidence map.
    n: int
        Window size (3,5,7,...)
    th: float
        Threshold for absolute difference
    Returns
    -------
    conf_rst: HxW np.ndarray
        Binary confidence map (1 for rejected points)
    """
    h, w = dmap.shape[:2]

    delta_map = np.zeros(dmap.shape, dtype=np.float32)

    nx = (nx - 1) // 2
    ny = (ny - 1) // 2

    # Conti's filtering method
    for y in range(h):
        for x in range(w):
            if dmap[y, x] > 0:
                # Search min
                dmin = 1000000.0
                for yw in range(-ny, ny + 1):
                    for xw in range(-nx, nx + 1):
                        if 0 <= y + yw and y + yw <= h - 1 and 0 <= x + xw and x + xw <= w - 1:
                            if dmap[y + yw, x + xw] < dmin and dmap[y + yw, x + xw] > 1e-3:
                                dmin = dmap[y + yw, x + xw]

                # Find pixel-wise confidence
                for yw in range(-ny, ny + 1):
                    for xw in range(-nx, nx + 1):
                        if 0 <= y + yw and y + yw <= h - 1 and 0 <= x + xw and x + xw <= w - 1:
                            if delta_map[y + yw, x + xw] < dmap[y + yw, x + xw] - dmin:
                                delta_map[y + yw, x + xw] = dmap[y + yw, x + xw] - dmin

    return delta_map


def filter_heuristic_depth(dmap, nx=7, ny=3, th=1.5, th_filter=0.1):
    dmap_copy = dmap.copy()
    deltamap = delta_depth(dmap_copy, nx, ny)
    conf_map = conti_conf_depth(deltamap, th)
    _ = filter(dmap_copy, conf_map, th_filter)
    return dmap_copy, conf_map


if __name__ == "__main__":
    # example of how to get a sparse depth map from a dense GT depth map
    dense_depth = np.random.randint(0, 5, (10, 10))
    valid_mask = dense_depth > 0
    hints_mask = deterministic_select_k_true(valid_mask, 50, 2024)
    sparse_depth = np.zeros_like(dense_depth)
    sparse_depth[hints_mask] = dense_depth[hints_mask]
    print("Sparse depth: \n", sparse_depth, "\n")

    # example of how to filter using a 7x7 window, as stated in section 4.1 in the paper
    sparse_depth, _ = filter_heuristic_depth(sparse_depth, nx=7, ny=7, th=1.5)
    hints_mask = sparse_depth > 0
    sparse_depth[hints_mask] = dense_depth[hints_mask]
    print("Filtered sparse depth: \n", sparse_depth)
