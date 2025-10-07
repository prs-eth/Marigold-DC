# Dataset Processing Instructions

This document provides instructions for downloading and processing evaluation datasets used in Marigold-DC, especially for those that don't come with sparse depth maps.

First, set the script directory variable.

```bash
export SCRIPT_DIR=<YOUR_SCRIPT_DIR>  # e.g., ~/Marigold-DC/script
```

For each dataset, download and process using the [dataset_processing.py](script/dataset_processing.py) script, passing the appropriate dataset flag.

## iBims-1

We process all 100 available images in the iBims-1 dataset at original resolution 640 × 480, sampling 1000 random depth points from the intersection of valid pixel masks (invalid, transparent, missing).

Download the [iBims-1](https://mediatum.ub.tum.de/1455541) dataset in your BASE_DATA_DIR directory:

```bash
cd $BASE_DATA_DIR
rsync --progress rsync://m1455541@dataserv.ub.tum.de/m1455541/ibims1_core_mat.zip .
# password is m1455541
unzip -d ibims1_core_mat ibims1_core_mat.zip && rm ibims1_core_mat.zip
```

After that, call the `dataset_processing.py` script for iBims-1.

```bash
cd $SCRIPT_DIR
python dataset_processing.py --dataset ibims1
# clean up
rm -r $BASE_DATA_DIR/ibims1_core_mat/
```

## VOID

We utilize all 800 frames from the 8 designated test sequences, and the provided sparse depth maps with three density levels of 150, 500, and 1500 points. Inference is performed at original resolution of 640 × 480.

The [VOID dataset](https://github.com/alexklwong/void-dataset) can be downloaded following the instruction in the repository:

```bash
cd $BASE_DATA_DIR 
wget -O void_release.zip 'https://yaleedu-my.sharepoint.com/:u:/g/personal/alex_wong_yale_edu/Ebwvk0Ji8HhNinmAcKI5vSkBEjJTIWlA8PXwKNQX_FvB7g?e=0Zqe7g&download=1'
unzip void_release.zip && rm void_release.zip
```

After that, call the `dataset_processing.py` script for VOID.

```bash
cd $SCRIPT_DIR
python dataset_processing.py --dataset void
# clean up
rm -r $BASE_DATA_DIR/void_release/
```

## NYU-Depth V2

We evaluate on the original test split consisting of 654 samples.
Images are downsampled to 320 × 240 and then center-cropped to 304 × 228, following established practice.
The sparse depth input is 500 random points.

We use the preprocessed NYUv2 HDF5 dataset provided by [Andrea Conti](https://github.com/andreaconti/sparsity-agnostic-depth-completion). Download with:

```bash
cd $BASE_DATA_DIR
wget https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/nyu_img_gt.h5
wget https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/nyu_pred_with_500.h5
```

After that, call the `dataset_processing.py` script for NYU-Depth V2.

```bash
cd $SCRIPT_DIR
python dataset_processing.py --dataset nyu
# clean up
rm $BASE_DATA_DIR/nyu_img_gt.h5
rm $BASE_DATA_DIR/nyu_pred_with_500.h5
```

NOTE: We process images at 768 resolution via upscaling at inference time, since otherwise the latent would be too small. Guidance is still performed at 304 × 228.

## KITTI DC

We evaluate on the original validation split consisting of 1000 samples, processing at the original resolution of 1216 × 352
The sparse depth input is filtered like in [VPP4DC](https://github.com/bartn8/vppdc/).

You can download the KITTI DC validation split from the [official website](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion). You can also directly download it:

```bash
cd $BASE_DATA_DIR
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip
unzip data_depth_selection.zip && rm data_depth_selection.zip
```

After that, call the `dataset_processing.py` script for KITTI DC. This will copy over the data in the expected format and filter the sparse depth using the local window method (sec 4.1 of the paper).

```bash
cd $SCRIPT_DIR
python dataset_processing.py --dataset kittidc
# clean up
rm -r $BASE_DATA_DIR/depth_selection/
```

## DDAD

We use the official DDAD val split, which has 3950 samples (front-view only). Images have a resolution of 1936 × 1216, but we perform inference at a processing resolution of 768 to keep memory usage manageable.

We use the dataset pre-processed by the VPP4DC authors. You can download with gdown:
```bash
cd $BASE_DATA_DIR
gdown 1y8Rt3Hld8zVTSKxx9d9yYXSzr5niKN7i
unzip ddad_pregenerated.zip && rm ddad_pregenerated.zip
```

The dataset is otherwise available on Drive here:
```
https://drive.google.com/open?id=1y8Rt3Hld8zVTSKxx9d9yYXSzr5niKN7i
```

After that, call the `dataset_processing.py` script for DDAD. This will copy over the data in the expected format and filter the sparse depth using the local window method:

```bash
cd $SCRIPT_DIR
python dataset_processing.py --dataset ddad
# clean up
rm -r $BASE_DATA_DIR/pregenerated/
```