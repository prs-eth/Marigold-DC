"""
This script computes metrics between predicted depth maps and ground truth depth maps.
It outputs both individual file metrics and global averages in CSV format.

The data_dir should have the following structure:
data_dir/
└── gt/                 # Ground truth depth arrays in meters (.npy files)
    ├── image_001.npy
    └── ...

The prediction_dir should contain:
prediction/             # Predicted depth arrays in meters (.npy files)
├── image_001.npy
└── ...
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import validate_prediction_gt_structure


def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    pred_masked = pred[mask]
    gt_masked = gt[mask]

    mae = np.mean(np.abs(pred_masked - gt_masked))
    rmse = np.sqrt(np.mean((pred_masked - gt_masked) ** 2))

    return {"mae": mae, "rmse": rmse}


def evaluate_single_file(pred_path: str, gt_path: str) -> Dict[str, float]:
    """
    Evaluate a single prediction file against its ground truth, both in meters.
    Filepath must point to the .npy files. Returns evaluation metrics (mae, rmse).
    """
    try:
        pred = np.load(pred_path)
        gt = np.load(gt_path)

        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

        mask = gt > 0

        if not np.any(mask):
            logging.warning(f"No valid pixels found in {os.path.basename(pred_path)}")
            return {"mae": np.nan, "rmse": np.nan}

        metrics = compute_metrics(pred, gt, mask)

        return {"mae": metrics["mae"], "rmse": metrics["rmse"]}

    except Exception as e:
        logging.error(f"Error evaluating {os.path.basename(pred_path)}: {e}")
        return {"mae": np.nan, "rmse": np.nan}


def main():
    parser = argparse.ArgumentParser(description="Evaluate depth completion predictions")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory containing gt/ subdirectory")
    parser.add_argument("--prediction_dir", type=str, required=True, help="Directory containing prediction .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for evaluation results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    gt_dir = os.path.join(args.data_dir, "gt")

    success, error_msg = validate_prediction_gt_structure(args.prediction_dir, gt_dir)
    if not success:
        logging.error(f"Folder structure is not valid: {error_msg}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    prediction_files = sorted([f for f in Path(args.prediction_dir).glob("*.npy")])
    gt_files = sorted([f for f in Path(gt_dir).glob("*.npy")])

    results = []
    for pred_path, gt_path in tqdm(zip(prediction_files, gt_files), total=len(prediction_files)):
        metrics = evaluate_single_file(pred_path, gt_path)
        results.append({"filename": os.path.basename(pred_path), "mae": metrics["mae"], "rmse": metrics["rmse"]})

    df = pd.DataFrame(results)

    global_mae = df["mae"].mean()
    global_rmse = df["rmse"].mean()
    global_row = pd.DataFrame({"filename": ["GLOBAL_AVERAGE"], "mae": [global_mae], "rmse": [global_rmse]})
    df_with_global = pd.concat([global_row, df], ignore_index=True)

    csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
    df_with_global.to_csv(csv_path, index=False)

    logging.info("=" * 50)
    logging.info("EVALUATION COMPLETE")
    logging.info("=" * 50)
    logging.info(f"Total files evaluated: {len(prediction_files)}")
    logging.info(f"Global MAE: {global_mae:.3f}")
    logging.info(f"Global RMSE: {global_rmse:.3f}")
    logging.info(f"Results saved to: {csv_path}")

    return 0


if __name__ == "__main__":
    main()
