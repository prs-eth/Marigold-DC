import argparse
import os
import shutil
from glob import glob

import h5py
import numpy as np
from PIL import Image
from scipy import io
from tqdm import tqdm
from utils import deterministic_select_k_true, filter_heuristic_depth


def process_nyu():
    """
    Process the NYU-Depth V2 dataset with sparse depth with 500 non-zero values.
    """
    base_data_dir = os.getenv("BASE_DATA_DIR")
    if base_data_dir is None:
        print("Error: BASE_DATA_DIR environment variable is not set")
        exit(1)
    output_path = os.path.join(base_data_dir, "nyudepthv2")
    for subdir in ["gt", "sparse", "rgb", "intrinsics"]:
        os.makedirs(os.path.join(output_path, subdir), exist_ok=True)

    hf_img = h5py.File(os.path.join(base_data_dir, "nyu_img_gt.h5"), "r")
    hf = h5py.File(os.path.join(base_data_dir, "nyu_pred_with_500.h5"), "r")

    camera_matrix = np.array([[5.1885790117450188e02 / 2.0, 0, 3.2558244941119034e02 / 2.0 - 8.0], [0, 5.1946961112127485e02 / 2.0, 2.5373616633400465e02 / 2.0 - 6.0], [0, 0, 1.0]])

    for index in tqdm(range(len(hf_img["gt"]))):
        index_str = f"{index:04d}"
        gt_depth = hf_img["gt"][index].squeeze()
        rgb = hf_img["img"][index]
        sparse_depth = hf["hints"][index].squeeze()

        rgb_image_path = os.path.join(output_path, "rgb", f"{index_str}.png")
        rgb = Image.fromarray(rgb)
        rgb.save(rgb_image_path, "PNG")

        sparse_depth_path = os.path.join(output_path, "sparse", f"{index_str}.npy")
        np.save(sparse_depth_path, sparse_depth)

        gt_depth_path = os.path.join(output_path, "gt", f"{index_str}.npy")
        np.save(gt_depth_path, gt_depth)

        np.savetxt(os.path.join(output_path, "intrinsics", f"{index_str}.txt"), camera_matrix)
        if index == 0:
            print("Debugging sample:")
            print(f"RGB shape: {np.array(rgb).shape}")
            print(f"Sparse depth shape: {sparse_depth.shape}")
            print(f"Number of sparse depth points: {(sparse_depth > 0).sum()}")
            print(f"GT depth shape: {gt_depth.shape}")
            print(f"GT depth range: [{gt_depth[gt_depth > 0].min():.3f}, {gt_depth[gt_depth > 0].max():.3f}]")

    print("NYU-Depth V2 dataset processed successfully at", output_path)


def process_ibims1():
    """
    Process the iBims-1 dataset creating sparse depth with 1000 non-zero values.
    """
    base_data_dir = os.getenv("BASE_DATA_DIR")
    if base_data_dir is None:
        print("Error: BASE_DATA_DIR environment variable is not set")
        exit(1)
    output_path = os.path.join(base_data_dir, "ibims1")
    for subdir in ["gt", "sparse", "rgb", "intrinsics"]:
        os.makedirs(os.path.join(output_path, subdir), exist_ok=True)

    basenames_list = [x.replace(".mat", "") for x in os.listdir(os.path.join(base_data_dir, "ibims1_core_mat", "ibims1_core_mat"))]
    for idx, basename in tqdm(enumerate(basenames_list), total=len(basenames_list)):
        image_data = io.loadmat(os.path.join(base_data_dir, "ibims1_core_mat", "ibims1_core_mat", basename + ".mat"))
        data = image_data["data"]

        rgb = data["rgb"][0][0]  # RGB image
        depth = data["depth"][0][0]  # Raw depth map
        calib = data["calib"][0][0]  # Calibration parameters
        mask_invalid = data["mask_invalid"][0][0]  # Mask for invalid pixels
        mask_transp = data["mask_transp"][0][0]  # Mask for transparent pixels

        mask_missing = depth.copy()
        mask_missing[mask_missing != 0] = 1  # mask for non-zero depth values

        mask_valid = mask_transp * mask_invalid * mask_missing  # Combine masks, section 4.1 in the paper
        mask_valid = mask_valid.astype(bool)

        gt_depth = depth * mask_valid
        np.save(os.path.join(output_path, "gt", f"{basename}.npy"), gt_depth)

        hints_mask = deterministic_select_k_true(mask_valid, 1000, 2024)
        hints = np.zeros_like(gt_depth)
        hints[hints_mask] = gt_depth[hints_mask]
        np.save(os.path.join(output_path, "sparse", f"{basename}.npy"), hints)

        Image.fromarray(rgb).save(os.path.join(output_path, "rgb", f"{basename}.png"))

        np.savetxt(os.path.join(output_path, "intrinsics", f"{basename}.txt"), calib.T)
        if idx == 0:
            print("Debugging sample:")
            print(f"RGB shape: {rgb.shape}")
            print(f"Sparse depth shape: {hints.shape}")
            print(f"Number of sparse depth points: {(hints > 0).sum()}")
            print(f"GT depth shape: {gt_depth.shape}")
            print(f"GT depth range: [{gt_depth[gt_depth > 0].min():.3f}, {gt_depth[gt_depth > 0].max():.3f}]")

    print("iBims-1 dataset processed successfully at", output_path)


def process_kittidc():
    """
    Process the KITTI DC dataset filtering the sparse depth with a heuristic.
    """
    base_data_dir = os.getenv("BASE_DATA_DIR")
    if base_data_dir is None:
        print("Error: BASE_DATA_DIR environment variable is not set")
    output_path = os.path.join(base_data_dir, "kittidc")
    for subdir in ["gt", "sparse", "rgb", "intrinsics"]:
        os.makedirs(os.path.join(output_path, subdir), exist_ok=True)

    datapath = os.path.join(base_data_dir, "depth_selection", "val_selection_cropped")

    image_list = sorted(glob(os.path.join(datapath, "image/*.png")))
    gt_list = sorted(glob(os.path.join(datapath, "groundtruth_depth/*.png")))
    hints_list = sorted(glob(os.path.join(datapath, "velodyne_raw/*.png")))
    calibtxt_list = sorted(glob(os.path.join(datapath, "intrinsics/*.txt")))

    for id in tqdm(range(len(image_list))):
        depth = np.asarray(Image.open(gt_list[id])) / 256

        np.save(os.path.join(output_path, "gt", f"{id:04d}.npy"), depth)

        # filter the sparse depth with a heuristic
        hints = np.asarray(Image.open(hints_list[id])) / 256
        hints, _ = filter_heuristic_depth(hints, nx=7, ny=7, th=1.5)
        np.save(os.path.join(output_path, "sparse", f"{id:04d}.npy"), hints)

        rgb = Image.open(image_list[id])
        rgb_image_path = os.path.join(output_path, "rgb", f"{id:04d}.png")
        rgb.save(rgb_image_path, "PNG")

        shutil.copy(calibtxt_list[id], os.path.join(output_path, "intrinsics", f"{id:04d}.txt"))
        if id == 0:
            print("Debugging sample:")
            print(f"RGB shape: {np.array(rgb).shape}")
            print(f"Sparse depth shape: {hints.shape}")
            print(f"Number of sparse depth points: {(hints > 0).sum()}")
            print(f"GT depth shape: {depth.shape}")
            print(f"GT depth range: [{depth[depth > 0].min():.3f}, {depth[depth > 0].max():.3f}]")

    print("KITTI DC dataset processed successfully at", output_path)


def process_ddad():
    """
    Process the DDAD dataset filtering the sparse depth with a heuristic.
    """
    base_data_dir = os.getenv("BASE_DATA_DIR")
    if base_data_dir is None:
        print("Error: BASE_DATA_DIR environment variable is not set")
        exit(1)
    output_path = os.path.join(base_data_dir, "ddad")
    for subdir in ["gt", "sparse", "rgb", "intrinsics"]:
        os.makedirs(os.path.join(output_path, subdir), exist_ok=True)
    dataset_path = os.path.join(base_data_dir, "pregenerated", "val")

    for id in tqdm(range(len(sorted(os.listdir(os.path.join(dataset_path, "gt")))))):
        depth = np.asarray(Image.open(os.path.join(dataset_path, "gt", f"{id:010d}.png"))) / 256
        np.save(os.path.join(output_path, "gt", f"{id:04d}.npy"), depth)

        hints = np.asarray(Image.open(os.path.join(dataset_path, "hints", f"{id:010d}.png"))) / 256
        hints, _ = filter_heuristic_depth(hints, nx=7, ny=7, th=1.5)
        np.save(os.path.join(output_path, "sparse", f"{id:04d}.npy"), hints)

        rgb = Image.open(os.path.join(dataset_path, "rgb", f"{id:010d}.png"))
        rgb_image_path = os.path.join(output_path, "rgb", f"{id:04d}.png")
        rgb.save(rgb_image_path, "PNG")

        shutil.copy(os.path.join(dataset_path, "intrinsics", f"{id:010d}.txt"), os.path.join(output_path, "intrinsics", f"{id:04d}.txt"))
        if id == 0:
            print("Debugging sample:")
            print(f"RGB shape: {np.array(rgb).shape}")
            print(f"Sparse depth shape: {hints.shape}")
            print(f"Number of sparse depth points: {(hints > 0).sum()}")
            print(f"GT depth shape: {depth.shape}")
            print(f"GT depth range: [{depth[depth > 0].min():.3f}, {depth[depth > 0].max():.3f}]")
        if id == 10:
            break
    print("DDAD dataset processed successfully at", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to process")
    args = parser.parse_args()

    if args.dataset == "ibims1":
        process_ibims1()
    elif args.dataset == "nyu":
        process_nyu()
    elif args.dataset == "kittidc":
        process_kittidc()
    elif args.dataset == "ddad":
        process_ddad()
    else:
        print("Invalid dataset")
        exit(1)
