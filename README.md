# Ultra-Low Dose PET Challenge 2024

[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html)
[![Python](https://img.shields.io/badge/Python-3.13%2B-blue?logo=python)](https://www.python.org/)
[![Docs](https://img.shields.io/badge/docs-link-green)](docs/)


 This project supposed to be my submission and research playground for the **Ultra-Low Dose PET (ULDPET) Image Reconstruction Challenge**, which aims to improve the quality of PET images acquired with significantly reduced radiotracer dose. Unfortunately, due to some misfortunate events, I could not finish and submit my work. Nevertheless, I am back to building it to discover if what I thought will work or not. This is a research-oriented project, with which I want to explore and think about novel deep learning strategies for denoising ultra-low-dose PET scans using hybrid convolutional kernels and interleaved depth-wise separable convolutions (will explain this later) with some sprinkle of FiLM. 🚀


## 🧠 Motivation

- PET imaging is crucial in clinical diagnostics, but the radioactive tracers impose health risks and limit scan frequency.
- The goal is to **reconstruct high-quality images** from extremely noisy, ultra-low dose inputs.
- I believe we can further tailor neural architectures to exploit the inherent **spatial and statistical priors** of PET data more effectively than off-the-shelf networks.

## 🧪 Implemented Methods

- ✅ Custom U-Net with adaptive kernel sizes
- ✅ Deep supervision with hierarchical loss
- ✅ Efficient dataset pipeline using **Parquet** and **Zarr**
- ✅ Multi-GPU and mixed precision training via PyTorch Lightning
- ✅ On-the-fly synthetic degradation for data augmentation

## 🧪 Planned Experiments

- [ ] **Depth-wise separable convolutions** to reduce parameter count without losing spatial sensitivity.
- [ ] **Interleaving Depth-wise separable convolutions** for recovering cut information flow among separated groups  (I vectorized this but not sure if pytorch will be able to optimize this further)
- [ ] **Mixture of Kernels** different kernels as 'experts' with gating network acting on meta-data
- [ ] **Gradient-based augmentation** using spatial image derivatives to highlight important regions.
- [ ] **Adaptive loss functions** that prioritize detail restoration in high-uncertainty zones.
- [ ] **Zarr + Dask pipeline** for scalable I/O and distributed training.
- [ ] **Noise injection during training** for robustness and uncertainty calibration.

## 🧬 Data Preprocessing

- Preprocessing supports DICOM and IMA series using `SimpleITK`, with optional reorientation and normalization.
- Volumes are normalized to `[0, 65535]` and saved in efficient NumPy or Zarr formats.
- Parallel loading and memory monitoring are implemented via multiprocessing.

## 📦 Project Structure
```
GC_ULDPET_2024/
├── tests
|   ├── test_depth_wise.py
|   ├── test_unets.py
|   └── test_variations.py
├── unets.py               # UNet and FiLM-UNet implementations
├── upscaler.py            # Custom upsampling modules
├── layers.py              # NConv, FiLM intercepts, and encoder blocks
├── variations.py          # Experimental kernel/stride/dilation configurations
├── depth_wise.py          # Depthwise conv modules (in progress)
├── TODO.md                # Research and development task list
└── README.md
```
---

## 💡 Experimental & Planned Ideas

These are promising directions to be explored next:

- 🔄 **Mixture of kernels**: Combining standard, dilated, and separable convolutions in parallel per layer.
- 🔀 **Interleaved depth-wise convolution layers**: Overlapping volumetric slices to enhance spatial-spectral mixing.
- 🧼 **Learnable pooling**: Replace max/avg pooling with pooling functions chosen and learned network.
- 🧠 **Information bottleneck-based denoising**: Investigating variational dropout to retain only essential signals.
- 🧊 **Progressive training with cold-start curriculum**: Start with noise-free patches and increase noise gradually. Explore variations of transitions from soft to hard gating. 

## 🤝 Contributing

Contributions are welcome — especially around the experimental ideas!
If you'd like to contribute or implement one of these ideas, see the [Contribution Guide](#🤝-contributing). Then pick an idea from the 💡 Planned Ideas section and start hacking!

## 📊 Evaluation
I will include:
- Visual comparison plots
- Per-region performance analysis
- Runtime and memory profiling

## ⚠️ Known Issues
- Currently requires inputs with dimensions divisible by 16 (due to max pooling).
- FiLM layers may break if metadata shape mismatches batch size.
- Padding assumes small differences in shape (±1) between tensors.


## 📁 Dataset & Preprocessing
I currently focused on
- Structural preservation (via SSIM filtering)
- Spatial normalization (rotation handling)
- Volume construction
- Efficient data storage

### Raw Data Format:
The dataset originates from PET scans in .ima format. These are converted and organized into efficient columnar storage using Apache Parquet and Zarr to enable rapid access during training.

### Volume Construction:
Each patient sample is constructed as a 3D volume, preserving spatial resolution and temporal consistency across slices.

### Rotation-Aware Preprocessing:
To address rotational variances that can arise during scanning:
- I normalize patient orientation across the dataset.
- I optionally apply rotation-invariant augmentations using quaternion-aligned interpolation.
- A canonical alignment step ensures all volumes are in a standard view frame (e.g., axial-facing supine).

### Structural Similarity-Based Filtering (SSIM):
For supervision, I employ SSIM-based loss functions that encourage structural preservation:
- Used to compute frame-wise similarity against high-dose ground truth.
- Also used to filter noisy samples during preprocessing (low-SSIM inputs flagged for exclusion or augmentation).

###	Intensity Normalization:
I plan to add: 
- Global z-score or percentile-based clipping per volume.
- Histogram equalization considered for inter-patient consistency.

###	Pretraining-Friendly Packaging:
- Datasets are stored in shardable Parquet or Zarr chunks to enable streaming, lazy loading, and parallel processing during training.
- Meta-information (e.g., patient ID, dose level, slice range) is attached to each sample via Parquet metadata or Zarr attributes.

Ultra-low dose PET scans and full-dose counterparts are converted and preprocessed as follows:

- Stored in **Zarr arrays** for efficient parallel access
- Metadata and slice-level information managed in **Parquet**
- Dataset split into train/val/test using a deterministic hash of subject IDs


## 📄 License
This project is licensed under the terms of the [GNU General Public License v2.0](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html).