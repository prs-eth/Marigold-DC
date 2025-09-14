"""
This script runs inference on an entire dataset organized with the following structure:

input_dir/
├── rgb/                # RGB images (png, jpg, or jpeg)
│   ├── image_001.png
│   └── ...
├── sparse/             # Sparse depth arrays in meters (.npy files), null values have 0
│   ├── image_001.npy
│   └── ...
└── gt/                 # Ground truth dense depth arrays in meters (.npy files)
    ├── image_001.npy   # gt/ folder is optional for inference only, but required for evaluation later
    └── ...

Filenames must match between directories. Therefore, the script first validates the directory structure
of rgb and sparse directories and then runs inference on the full dataset.
"""

import argparse
import glob
import logging
import os
import sys
import warnings

import diffusers
import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image
from tqdm import tqdm
from utils import validate_rgb_sparse_structure

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from marigold_dc import MarigoldDepthCompletionPipeline

warnings.simplefilter(action="ignore", category=FutureWarning)
diffusers.utils.logging.disable_progress_bar()

def main():
    parser = argparse.ArgumentParser(description="Marigold-DC Dataset Inference")

    DEPTH_CHECKPOINT = "prs-eth/marigold-depth-v1-0"
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing rgb/, sparse/, and optionally gt/ subdirectories")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for predictions")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--ensemble_size", type=int, default=1, help="Number of predictions to be ensembled")
    parser.add_argument("--processing_resolution", type=int, default=768, help="Denoising resolution")
    parser.add_argument("--checkpoint", type=str, default=DEPTH_CHECKPOINT, help="Depth checkpoint")
    parser.add_argument("--seed", type=int, default=2024, help="Seed")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    if os.path.exists(args.output_dir):
        response = input(f"Output directory '{args.output_dir}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            logging.info("Operation cancelled by user")
            return 0

    success, error_msg = validate_rgb_sparse_structure(args.input_dir)
    if not success:
        logging.error(f"Folder structure is not valid: {error_msg}")
        return 1

    num_inference_steps = args.num_inference_steps
    ensemble_size = args.ensemble_size
    processing_resolution = args.processing_resolution
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        processing_resolution_non_cuda = 512
        num_inference_steps_non_cuda = 10
        ensemble_size_non_cuda = 1
        if processing_resolution > processing_resolution_non_cuda:
            logging.warning(f"CUDA not found: Reducing processing_resolution to {processing_resolution_non_cuda}")
            processing_resolution = processing_resolution_non_cuda
        if num_inference_steps > num_inference_steps_non_cuda:
            logging.warning(f"CUDA not found: Reducing num_inference_steps to {num_inference_steps_non_cuda}")
            num_inference_steps = num_inference_steps_non_cuda
        if ensemble_size > ensemble_size_non_cuda:
            logging.warning(f"CUDA not found: Reducing ensemble_size to {ensemble_size_non_cuda}")
            ensemble_size = ensemble_size_non_cuda

    pipe = MarigoldDepthCompletionPipeline.from_pretrained(args.checkpoint, prediction_type="depth").to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    if not torch.cuda.is_available():
        logging.warning("CUDA not found: Using a lightweight VAE")
        del pipe.vae
        pipe.vae = diffusers.AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device)

    rgb_dir = os.path.join(args.input_dir, "rgb")
    rgb_file_paths = []
    for ext in ['.png', '.jpg', '.jpeg']:
        rgb_file_paths.extend(glob.glob(os.path.join(rgb_dir, f"*{ext}")))
    rgb_file_paths = sorted(rgb_file_paths)
    basenames = [os.path.splitext(os.path.basename(f))[0] for f in rgb_file_paths]

    logging.info(f"Starting inference on {len(basenames)} files...")

    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)
    prediction_dir = os.path.join(output_path, "prediction")
    os.makedirs(prediction_dir, exist_ok=True)
    visuals_dir = os.path.join(output_path, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)

    successful_inferences = 0
    failed_inferences = 0

    with tqdm(total=len(basenames), desc="Running inference") as pbar:
        for rgb_image_path, basename in zip(rgb_file_paths, basenames):
            try:
                sparse_path = os.path.join(args.input_dir, "sparse", f"{basename}.npy")

                pred = pipe(
                    image=Image.open(rgb_image_path),
                    sparse_depth=np.load(sparse_path),
                    num_inference_steps=num_inference_steps,
                    ensemble_size=ensemble_size,
                    processing_resolution=processing_resolution,
                    seed=args.seed
                )

                pred_filepath = os.path.join(prediction_dir, f"{basename}.npy")
                np.save(pred_filepath, pred)

                vis_filepath = os.path.join(visuals_dir, f"{basename}_vis.jpg")
                vis = pipe.image_processor.visualize_depth(pred, val_min=pred.min(), val_max=pred.max())[0]
                vis.save(vis_filepath)

                successful_inferences += 1
                pbar.set_postfix({
                    'Success': successful_inferences,
                    'Failed': failed_inferences,
                    'Current': basename
                })

            except Exception as e:
                logging.error(f"Failed to process {basename}: {e}")
                failed_inferences += 1
                pbar.set_postfix({
                    'Success': successful_inferences,
                    'Failed': failed_inferences,
                    'Current': basename
                })

            pbar.update(1)

    # Summary
    logging.info("=" * 50)
    logging.info("INFERENCE COMPLETE")
    logging.info("=" * 50)
    logging.info(f"Total files processed: {len(basenames)}")
    logging.info(f"Successful inferences: {successful_inferences}")
    logging.info(f"Failed inferences: {failed_inferences}")
    logging.info(f"Output directory: {output_path}")

    if failed_inferences > 0:
        logging.warning(f"{failed_inferences} files failed to process. Check logs for details.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
