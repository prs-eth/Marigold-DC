#!/usr/bin/env python
"""
Basic script to download ScanNet .sens files from the scene list used for testing,
and then extract RGB images, depth images, and intrinsics from ScanNet .sens files.
"""

import argparse
import os
import shutil
import struct
import tempfile
import urllib.request
import zlib

import cv2
import imageio
import numpy as np
import png
from tqdm import tqdm

BASE_URL = "http://kaldir.vc.in.tum.de/scannet/"
TOS_URL = BASE_URL + "ScanNet_TOS.pdf"

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


def download_file(url, out_file):
    """Download a single file"""
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    print(f"Downloading: {url}")
    fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
    f = os.fdopen(fh, "w")
    f.close()
    urllib.request.urlretrieve(url, out_file_tmp)
    os.rename(out_file_tmp, out_file)
    print(f"Saved to: {out_file}")


def download_scene(scene_id, out_dir):
    """Download .sens file for a scene"""
    # ScanNet test scenes use v2 .sens files
    url = f"{BASE_URL}v2/scans/{scene_id}/{scene_id}.sens"
    out_file = os.path.join(out_dir, scene_id, f"{scene_id}.sens")
    download_file(url, out_file)


class RGBDFrame:
    def load(self, file_handle):
        self.camera_to_world = np.asarray(struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = b"".join(struct.unpack("c" * self.color_size_bytes, file_handle.read(self.color_size_bytes)))
        self.depth_data = b"".join(struct.unpack("c" * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))

    def decompress_depth(self, compression_type):
        if compression_type == "zlib_ushort":
            return self.decompress_depth_zlib()
        else:
            raise ValueError(f"Unknown compression type: {compression_type}")

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == "jpeg":
            return self.decompress_color_jpeg()
        else:
            raise ValueError(f"Unknown compression type: {compression_type}")

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)


class SensorData:
    def __init__(self, filename):
        self.version = 4
        self.load(filename)

    def load(self, filename):
        with open(filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            assert self.version == version, f"Expected version {self.version}, got {version}"
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = b"".join(struct.unpack("c" * strlen, f.read(strlen))).decode("utf-8")
            self.intrinsic_color = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack("i", f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack("i", f.read(4))[0]]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            self.frames = []
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def export_depth_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f"Exporting {len(self.frames) // frame_skip} depth frames to {output_path}")
        for f in range(0, len(self.frames), frame_skip):
            depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
            if image_size is not None:
                depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
            # Write 16-bit PNG
            with open(os.path.join(output_path, f"{f:04d}.png"), "wb") as file:
                writer = png.Writer(width=depth.shape[1], height=depth.shape[0], bitdepth=16)
                depth_list = depth.reshape(-1, depth.shape[1]).tolist()
                writer.write(file, depth_list)

    def export_color_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f"Exporting {len(self.frames) // frame_skip} color frames to {output_path}")
        for f in range(0, len(self.frames), frame_skip):
            color = self.frames[f].decompress_color(self.color_compression_type)
            if image_size is not None:
                color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
            imageio.imwrite(os.path.join(output_path, f"{f:04d}.jpg"), color)

    def save_mat_to_file(self, matrix, filename):
        with open(filename, "w") as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt="%f")

    def export_poses(self, output_path, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f"Exporting {len(self.frames) // frame_skip} camera poses to {output_path}")
        for f in range(0, len(self.frames), frame_skip):
            self.save_mat_to_file(self.frames[f].camera_to_world, os.path.join(output_path, f"{f:04d}.txt"))

    def export_intrinsics(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f"Exporting camera intrinsics to {output_path}")
        self.save_mat_to_file(self.intrinsic_color, os.path.join(output_path, "intrinsic_color.txt"))
        self.save_mat_to_file(self.extrinsic_color, os.path.join(output_path, "extrinsic_color.txt"))
        self.save_mat_to_file(self.intrinsic_depth, os.path.join(output_path, "intrinsic_depth.txt"))
        self.save_mat_to_file(self.extrinsic_depth, os.path.join(output_path, "extrinsic_depth.txt"))


def process_scene(scene_path, output_base_path):
    """
    Process a single scene directory containing a .sens file
    """
    scene_name = os.path.basename(scene_path)
    sens_file = os.path.join(scene_path, f"{scene_name}.sens")

    if not os.path.exists(sens_file):
        print(f"Warning: .sens file not found for scene {scene_name}")
        return

    print(f"\nProcessing scene: {scene_name}")

    # Load sensor data
    sd = SensorData(sens_file)

    # Export data
    depth_output = os.path.join(output_base_path, scene_name, "depth")
    sd.export_depth_images(depth_output)

    color_output = os.path.join(output_base_path, scene_name, "color")
    sd.export_color_images(color_output)

    intrinsic_output = os.path.join(output_base_path, scene_name, "intrinsic")
    sd.export_intrinsics(intrinsic_output)


def main():
    parser = argparse.ArgumentParser(description="Download ScanNet .sens files from scene list")
    parser.add_argument("--scene_list", type=str, required=True, help="Path to file with scene list")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for extracted data")
    args = parser.parse_args()

    print("By pressing any key to continue you confirm that you have agreed to the ScanNet terms of use as described at:")
    print(TOS_URL)
    print("***")
    print("Press Enter to continue, or CTRL-C to exit.")
    input("")

    # Read scene list
    with open(args.scene_list, "r") as f:
        scene_list = [line.strip() for line in f if line.strip()]
        # Extract unique scene names
        scenes = list(set([scene.split("/")[0] for scene in scene_list]))

    print(f"\nFound {len(scenes)} unique scenes to download")
    tmp_out_dir = os.path.join(args.output_dir, "tmp")
    print(f"Output directory: {tmp_out_dir}\n")

    # Download each scene
    downloaded_count = 0
    for scene_id in tqdm(scenes, desc="Downloading scenes"):
        try:
            download_scene(scene_id, tmp_out_dir)
            downloaded_count += 1
        except Exception as e:
            print(f"Error downloading {scene_id}: {e}")
            continue

    print(f"\n✓ Completed downloading {downloaded_count}/{len(scenes)} scenes")

    # Process each scene
    processed_count = 0
    for scene_dir in tqdm(scenes, desc="Processing scenes"):
        try:
            process_scene(os.path.join(tmp_out_dir, scene_dir), args.output_dir)
            processed_count += 1
        except Exception as e:
            print(f"Error processing scene {scene_dir}: {e}")
            continue

    print(f"\n✓ Completed processing {processed_count}/{len(scenes)} scenes")
    print(f"Data saved to {args.output_dir}")
    shutil.rmtree(tmp_out_dir)


if __name__ == "__main__":
    main()
