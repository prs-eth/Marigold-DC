# Dataset Processing Instructions

This document provides instructions for downloading and processing evaluation datasets used in Marigold-DC, especially for those that don't come with sparse depth maps.

First, set the script directory variable.

```bash
export SCRIPT_DIR=<YOUR_SCRIPT_DIR>  # e.g., ~/Marigold-DC/script
```

For each dataset, download and process using the [dataset_processing.py](script/dataset_processing.py) script, passing the appropriate dataset flag.

## 2. iBims-1

We process all 100 available images in the iBims-1 dataset at original resolution 640×480, sampling 1000 random depth points from the intersection of valid pixel masks (invalid, transparent, missing).

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
```

## 6. NYU-Depth V2

We evaluate on the original test split consisting of 654 samples.
Images are downsampled to 320×240 and then center-cropped to 304×228, following established practice.
The sparse depth input is 500 random points.

We use the preprocessed NYUv2 HDF5 dataset provided by [Andrea Conti](https://github.com/andreaconti/sparsity-agnostic-depth-completion). Download with:

```bash
cd $BASE_DATA_DIR
wget https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/nyu_img_gt.h5
wget https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/nyu_pred_with_500.h5
```

After that, call the `dataset_processing.py` script for iBims-1.

```bash
cd $SCRIPT_DIR
python dataset_processing.py --dataset ibims1
```

NOTE: We process images at 768 resolution via upscaling at inference time, since otherwise the latent would be too small. Guidance is still performed at 304×228.