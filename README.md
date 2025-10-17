# ⇆ Marigold-DC: Zero-Shot Monocular Depth Completion with Guided Diffusion (ICCV 2025)

[![Website](https://img.shields.io/badge/%F0%9F%A4%8D%20Project%20-Website-blue)](https://marigolddepthcompletion.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-PDF-b31b1b)](http://arxiv.org/abs/2412.13389)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Space-yellow)](https://huggingface.co/spaces/prs-eth/marigold-dc)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

This repository represents the official implementation of the paper titled "Marigold-DC: Zero-Shot Monocular Depth Completion with Guided Diffusion".

Photogrammetry and Remote Sensing team: 
[Massimiliano Viola](https://www.linkedin.com/in/massimiliano-viola/), 
[Kevin Qu](https://www.linkedin.com/in/kevin-qu-b3417621b/), 
[Nando Metzger](https://nandometzger.github.io/), 
[Bingxin Ke](http://www.kebingxin.com/),
[Alexander Becker](https://scholar.google.ch/citations?user=Wle2GmkAAAAJ&hl=en), 
[Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en),
[Anton Obukhov](https://www.obukhov.ai/).

![](doc/teaser.jpg)

## 🛠️ Setup

📦 Clone the repository:
```bash
git clone https://github.com/prs-eth/Marigold-DC.git
cd Marigold-DC
```

🐍 Create python environment:
```bash
python -m venv venv/marigold_dc
```

⚡ Activate the environment:
```bash
source venv/marigold_dc/bin/activate
```

💻 Install the dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

The script performs densification of the input sparse depth, provided as a sparse numpy array, 
and saves the output as a dense numpy array, along with the visualization. 
Optimal default settings are applied.
By default, it processes the [teaser image](data/image.png) and uses [100-point guidance](data/sparse_100.npy). 

🏃🏻‍♂️‍➡️ Simply run as follows:
```bash
python -m marigold_dc
```

🧩 Customize image and sparse depth inputs as follows:
```bash
python -m marigold_dc \
    --in-image <PATH_RGB_IMAGE> \
    --in-depth <PATH_SPARSE_DEPTH> \
    --out-depth <PATH_DENSE_DEPTH>
```

🛠️ Customize other settings:
- `--num_inference_steps <int>` specifies the number of diffusion inference steps (default: 50).
- `--ensemble_size <int>` specifies the number of predictions to be ensembled (default: 1).
- `--processing_resolution <int>` specifies the processing resolution for the denoising process (default: 768. Using 0 means processing at original resolution).
- `--checkpoint <path>` allows overriding the base monocular depth estimation model checkpoint; can be a local path or a Hugging Face repository.

## 🏋️‍♂️ Training

None — the method is purely test-time; please refer to the paper for more details.

## ⬇ Checkpoint cache
By default, the [checkpoint](https://huggingface.co/prs-eth/marigold-depth-v1-0) is stored in the Hugging Face cache, 
which defaults to the home directory on Linux and Mac. 
This is often problematic in cluster environments.
The `HF_HOME` environment variable defines the cache location and can be overridden, e.g.:

```
export HF_HOME=/large_volume/cache
```

## 🦿 Evaluation on test datasets

Set the data directory variable (needed in evaluation scripts) and download the evaluation datasets there, following the instructions in [DATASETS.md](DATASETS.md) to create the sparse depth maps in a reproducible way.

```bash
export BASE_DATA_DIR=<YOUR_DATA_DIR>  # e.g., ~/Marigold-DC/datasets/
```

Each dataset in the data directory should have the following format:
```
dataset_name/
├── rgb/                # RGB images (png, jpg, or jpeg)
│   ├── image_001.png
│   └── ...
├── sparse/             # Sparse depth arrays in meters (.npy files), null values have 0
│   ├── image_001.npy
│   └── ...
└── gt/                 # Ground truth dense depth arrays in meters (.npy files)
    ├── image_001.npy
    └── ...
```

Run inference and evaluation scripts, for example:
```bash
# Scannet
bash script/eval/11_infer_scannet.sh  # Run inference
bash script/eval/12_eval_scannet.sh   # Evaluate predictions
```

All scripts with the correct inference parameters are available in the `script/eval/` directory.

## 🏎️💨 Inference speed

By default, the code runs with `bfloat16` precision on supported GPUs, enabling faster inference with negligible accuracy loss while preserving sufficient gradient precision for backpropagation.
Full-precision inference in `float32` can be re-enabled by specifying the `--use_full_precision` flag.

In addition, a lightweight [Tiny VAE](https://github.com/madebyollin/taesd) (the suggested option for CPU processing) can also be enabled on GPU by setting the `--use_tiny_vae` flag. Note that this comes at the cost of prediction quality.

Compiling the model can further improve inference speed, but again at the cost of performance. This is most beneficial when the same pipeline instance is used repeatedly, and can be achieved by calling `torch.compile` after the pipeline has been loaded:

```python
pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
```

Below, we report average runtime on A100 and performance on the 100 samples from NYUv2 used for ablation in the main paper.

| Variant                        | Time (50 Steps) | Speed        | MAE   | RMSE  |
|--------------------------------|---------------:|------------:|------:|------:|
| **float32**                    | 18.03 sec      | 2.77 iter/s  | 0.066 | 0.171 |
| **bfloat16** (*)               | 10.71 sec      | 4.67 iter/s  | 0.066 | 0.171 |
| **bfloat16 + compile**         | 8.85 sec       | 5.65 iter/s  | 0.067 | 0.172 |
| **Tiny VAE**                   | 9.99 sec       | 5.00 iter/s  | 0.068 | 0.174 |
| **TinyVAE + bfloat16**       | 5.63 sec       | 8.88 iter/s  | 0.069 | 0.173 |
| **TinyVAE + bfloat16 + compile** | 3.70 sec      | 13.53 iter/s | 0.070 | 0.175 |

(*) Used by default

## Abstract

Depth completion upgrades sparse depth measurements into dense depth maps, guided by a conventional image. 
Existing methods for this highly ill-posed task operate in tightly constrained settings, 
and tend to struggle when applied to images outside the training domain, 
as well as when the available depth measurements are sparse, irregularly distributed, or of varying density. 
Inspired by recent advances in monocular depth estimation, 
we reframe depth completion as image-conditional depth map generation, guided by a sparse set of measurements. 
Our method, Marigold-DC, builds on a pretrained latent diffusion model (LDM) for depth estimation and injects 
the depth observations as test-time guidance, via an optimization scheme that runs in tandem with the iterative 
inference of denoising diffusion. The method exhibits excellent zero-shot generalization across a diverse range 
of environments and handles even extremely sparse guidance effectively. Our results suggest that contemporary 
monodepth priors greatly robustify depth completion: it may be better to view the task as recovering dense depth 
from (dense) image pixels, guided by sparse depth; rather than as inpainting (sparse) depth, guided by an image.

## 📢 News

 - 2025-10-16: Added options to speed up inference.
 - 2025-10-08: Evaluation code is released.
 - 2025-07-23: The paper is accepted at ICCV 2025.
 - 2024-12-19: ArXiv paper and demo release.
 - 2024-12-18: Code release (this repository).

## 🎓 Citation
```bibtex
@misc{viola2024marigolddc,
    title={Marigold-DC: Zero-Shot Monocular Depth Completion with Guided Diffusion}, 
    author={Massimiliano Viola and Kevin Qu and Nando Metzger and Bingxin Ke and Alexander Becker and Konrad Schindler and Anton Obukhov},
    year={2024},
    eprint={2412.13389},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
}
```

## 🎫 License

The code of this work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).
