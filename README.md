# Teeth Segmentation — Tianchi 2023

<p align="center">
  <b>Multi-Model Ensemble for 2D Dental Panoramic X-ray Segmentation</b><br>
  <i>Tianchi Teeth Segmentation Competition · 2023</i>
</p>

---

This repository contains the complete training and inference pipeline for the **2D Teeth Segmentation** track of the 2023 Tianchi competition. The task is to produce pixel-level binary segmentation masks of teeth from dental panoramic X-ray images.

## Overview

Dental panoramic radiographs (OPGs) are widely used in clinical practice for diagnosis and treatment planning. Accurate, automated teeth segmentation from these images enables downstream applications such as tooth counting, anomaly detection, and orthodontic assessment.

The dataset consists of **2,000 training images** (320×640, 3-channel BGR) with corresponding binary masks, and **500 test images** without ground truth. The main challenges include:

- Large variation in tooth count, shape, and alignment across patients
- Low contrast at tooth boundaries, especially for overlapping or impacted teeth
- Presence of dental restorations, implants, and orthodontic appliances

## Technical Approach

### Architecture Overview

We implemented and compared **seven encoder-decoder architectures** spanning both CNN-based and Transformer-based backbones. The final submission is a 5-fold ensemble of the best-performing model.

| # | Config Key | Encoder | Decoder | Params | Input |
|---|------------|---------|---------|--------|-------|
| 1 | `SwinV1-Small` | Swin Transformer V1 Small | UPerNet | ~50M | 768 |
| 2 | `Segformer-b3` | SegFormer MIT-B3 | DAFormer (3×3) | ~45M | 768 |
| 3 | `pvt-unet` | PVT-V2-B4 | U-Net (SMP) | ~63M | 768 |
| 4 | `convnext-smpunet` | ConvNeXt-Large (ImageNet-22k→1k) | SMP U-Net | ~198M | 1280 |
| 5 | `coat-daformer` | CoAt-Parallel-Small | DAFormer (3×3) | ~24M | 768 |
| 6 | `dual-daformer` | DualViT-B | DAFormer (3×3) | ~42M | 768 |
| 7 | **`pvt-unet-mp`** | **PVT-V2-B4 + MeanPool** | **DAFormer (1×1)** | **~63M** | **768** |

> **Model selection**: Set `CFG.model` in [`Config.py`](code/Config.py) to the desired key before training.

### Best Model: PVT-V2-B4 + MeanPool + DAFormer

The top-performing model (`pvt-unet-mp`) uses a modified [Pyramid Vision Transformer V2](https://github.com/whai362/PVT) as the encoder, with a novel **MeanPool multi-scale aggregation** strategy:

```
Input (B, 3, H, W)  — BGR image, 3 channels
  │
  ├─ Split into K=3 single-channel views: [v[:,0:1], v[:,1:2], v[:,2:3]]
  ├─ Stack: (3B, 1, H, W)
  │
  ├─ Patch Embed (7×7, stride 4, 1ch→64ch)
  │
  ├─ Stage 1: [64ch,  ×3 blocks,  SR-ratio 8]   → H/4
  ├─ Stage 2: [128ch, ×8 blocks,  SR-ratio 4]   → H/8
  ├─ Stage 3: [320ch, ×27 blocks, SR-ratio 2]   → H/16
  └─ Stage 4: [512ch, ×3 blocks,  SR-ratio 1]   → H/32
        │
        │  MeanPool: reshape (3, B, C, h, w) → mean over K=3
        │  (fuses information from all BGR channels at each stage)
        │
  └─ DAFormer Decoder (1×1 fuse, 256ch)
        │
        ├─ MLP per stage (1×1 Conv + BN + ReLU + MixUpSample)
        ├─ Concat + Fuse (1×1 Conv + BN + ReLU)
        └─ 1×1 Conv → Logit → Bilinear Upsample to (H, W)
```

**MeanPool aggregation**: The 3-channel BGR input is split into K=3 single-channel views and processed independently through the PVT-V2-B4 encoder (whose patch embedding is adapted from 3ch to 1ch). At each encoder stage, the output is reshaped as `(K, B, C, h, w)` and averaged across the K dimension, fusing information from all color channels before passing to the decoder.

**DAFormer decoder**: Based on [Hoyer et al., CVPR 2022](https://github.com/lhoyer/DAFormer), each encoder feature is projected to a common dimension (256) via 1×1 convolutions, upsampled to a shared resolution using learned MixUp-sampling (bilinear + nearest interpolation with learned mixing weight), concatenated, and fused with a 1×1 convolution.

### Training Strategy

| Component | Setting |
|-----------|---------|
| **Cross-Validation** | 5-fold (KFold, `random_state=27`) |
| **Optimizer** | Lookahead (α=0.5, k=5) wrapping RAdam |
| **LR Schedule** | Linear decay over first 2/3 of epochs (56 ep), then constant at `min_lr` |
| **Learning Rate** | `1e-4` → `1e-6` (linear decay) |
| **Batch Size** | Train: 8, Validation: 16 |
| **Epochs** | 84 (checkpoints saved after epoch 42) |
| **Mixed Precision** | AMP (`torch.cuda.amp.autocast`) |
| **Multi-GPU** | `torch.nn.parallel.data_parallel` |

**LR Schedule detail**: The learning rate decays linearly from `start_lr` to `min_lr` over `epochs × 2/3 = 56` epochs, then remains constant at `min_lr` for the remaining 28 epochs:

```
lr(epoch) = max(min_lr, (56 - epoch) / 56 × (start_lr - min_lr) + min_lr)
```

### Loss Function

The training objective combines a main segmentation loss with an auxiliary deep supervision loss:

```
L_total = L_main + 0.2 × L_aux2
```

**Main loss** — two variants depending on the model:

| Models | Formula |
|--------|---------|
| `pvt-unet`, `pvt-unet-mp` | `0.7 × DiceLoss + 0.3 × FocalLoss` |
| `SwinV1-Small`, `Segformer-b3`, `coat-daformer`, `convnext-smpunet` | `0.7 × DiceLoss + 0.3 × BCE` |

**Auxiliary loss** (applied to encoder stage outputs for deep supervision):

```
L_aux_i = 0.7 × DiceLoss_i + 0.3 × FocalLoss_i
```

- **Dice Loss**: `1 - (2 × |P ∩ G| + ε) / (|P| + |G| + ε)`, where P is sigmoid prediction and G is ground truth
- **Focal Loss**: `α × (1 - p)^γ × BCE(logit, target)`, with α=0.25, γ=2 (from [Lin et al., ICCV 2017](https://arxiv.org/abs/1708.02002))

### Data Augmentation

Custom augmentation pipeline implemented in [`transforms_tc.py`](code/MyDataset/transforms_tc.py). All transforms operate on numpy arrays and are probability-gated:

| Transform | p | Parameters |
|-----------|---|------------|
| Random H/V Flip | 0.5 | Horizontal, vertical, or both (independent) |
| Random Rotate 90° | 0.5 | 0° / 90° / 180° / 270° (uniform) |
| Random Noise | 0.5 | Uniform noise, magnitude 0.1 |
| Random Contrast | 0.5 | Multiplicative factor α ∈ [0.6, 1.4] |
| Random Rotate + Scale | 0.5 | Angle ±30°, scale ∈ [0.5, 2.0], constant border fill |
| Random Cutout | 0.5 | 1–5 rectangular blocks, each 10–30% of image size |

### Preprocessing Pipeline

```
1. cv2.imread(path, IMREAD_COLOR)          # BGR, uint8, 320×640
2. Center-pad to 640×640                   # (pad_h, pad_w) = (160, 0)
3. cv2.resize(fx=CFG.scale, fy=CFG.scale)  # Optional scaling
4. Normalize: image / 255.0                # float32, [0, 1]
5. Mask binarize: mask[mask > 0] = 1       # float32, {0, 1}
```

### Post-Training: Stochastic Weight Averaging (SWA)

After training completes, the top-5 epoch checkpoints from each fold are averaged to produce a single SWA weight per fold. This smooths the loss landscape and typically improves generalization:

```python
# swa.py: for each fold, average 5 best checkpoints
swa_weights = mean([checkpoint_84, checkpoint_81, checkpoint_79, checkpoint_76, checkpoint_73])
```

### Inference: Test-Time Augmentation (TTA)

At inference time, each image is predicted **4 times** with rotational augmentation, then averaged:

| Pass | Transform | Undo |
|------|-----------|------|
| 1 | Original | — |
| 2 | `rot90(k=1)` | `rot90(k=-1)` |
| 3 | `rot90(k=2)` | `rot90(k=-2)` |
| 4 | `rot90(k=3)` | `rot90(k=-3)` |

The 4 probability maps are averaged, then thresholded at **0.55** to produce the final binary mask. Padding is removed by cropping `[160:-160, :]` to restore the original 320×640 shape.

### Ensemble Submission

The final submission ensembles **5 fold checkpoints** of the PVT-V2-B4 MeanPool model:

```
P_final = mean(P_fold0, P_fold1, P_fold2, P_fold3, P_fold4)   # each with TTA (×4)
mask = (P_final > 0.55).astype(uint8) × 255
```

This yields **20 forward passes** per test image (5 folds × 4 TTA rotations).

## Project Structure

```
tianchi-2023-teeth-segmentation/
├── README.md
├── LICENSE                              # MIT License
├── NOTICE                               # Third-party attributions
├── .gitignore
├── requirements.txt
├── code/                                # ← run all scripts from this directory
│   ├── Config.py                        # Global hyperparameter configuration
│   ├── arg_parser.py                    # CLI argument parser
│   ├── engine.py                        # Validation loop (Dice, IoU)
│   ├── submit.py                        # Inference + ensemble submission
│   ├── swa.py                           # Stochastic Weight Averaging utility
│   ├── run_train_pvtunet_mp.py          # ★ Train best model (PVT-V2-B4 + MeanPool)
│   ├── run_train_pvtunet.py             # Train PVT-V2-B4 + U-Net
│   ├── run_train_coat.py                # Train CoAt + DAFormer
│   ├── run_train_segformer.py           # Train SegFormer MIT-B3
│   ├── run_train_convnext_smpunet.py    # Train ConvNeXt + SMP U-Net
│   ├── run_train_dualvit_daformer.py    # Train DualViT + DAFormer
│   ├── run_train.py                     # Train Swin V1 + UPerNet
│   ├── run_train_all.py                 # Train on full data (no val split)
│   ├── data_preprocess.ipynb            # Data exploration notebook
│   ├── train.csv                        # 2,000 training samples index
│   ├── test.csv                         # 500 test samples index
│   ├── Model/                           # Model architecture definitions
│   │   ├── __init__.py                  #   build_model() dispatcher (7 models)
│   │   ├── pvt_v2.py / pvt_v2_2.py     #   Pyramid Vision Transformer V2 [Wang et al.]
│   │   ├── swin_v1_variable.py          #   Swin Transformer V1 [Liu et al.]
│   │   ├── coat.py                      #   CoAt-Net [Dai et al.]
│   │   ├── dualvit.py                   #   Dual Vision Transformer [Ren et al.]
│   │   ├── mit.py                       #   Mix Transformer [Xie et al.]
│   │   ├── daformer.py                  #   DAFormer Decoder [Hoyer et al.]
│   │   ├── upernet.py                   #   UPerNet Decoder [Xiao et al.]
│   │   ├── smp_unet.py                  #   SMP U-Net Decoder [Yakubovskiy]
│   │   ├── model.py                     #   Swin + UPerNet full model
│   │   ├── model_*_for_train.py         #   Full model defs (with loss computation)
│   │   └── model_*_for_submit.py        #   Full model defs (inference only)
│   ├── Libs/                            # Utility library
│   │   ├── libs.py                      #   LR scheduler, seeding, metrics, FocalLoss
│   │   ├── metrics.py                   #   Surface distance metrics [Google, Apache 2.0]
│   │   └── lookup_tables.py             #   Marching cubes tables [Google, Apache 2.0]
│   ├── MyDataset/                       # Data loading & augmentation
│   │   ├── __init__.py                  #   build_dataset() dispatcher
│   │   ├── dataset.py                   #   BaseDataset, CSV loading, fold splitting
│   │   └── transforms_tc.py             #   Custom augmentation transforms
│   ├── MyMetric/                        # Loss functions
│   │   └── __init__.py                  #   DiceLoss, BCE, FocalLoss, AuxLoss
│   └── MyOptimizer/                     # Optimizer
│       └── optim.py                     #   Lookahead wrapper [Zhang et al., 2019]
├── data/
│   └── README.md                        # Dataset download instructions
```

> Pretrained backbone weights should be placed in `code/pretrained/` (see Step 3 below).

## Quick Start

### 1. Environment Setup

```bash
# Python >= 3.8, CUDA >= 11.3
pip install -r requirements.txt
```

**Key dependencies**: PyTorch ≥ 1.12, timm ≥ 0.6, segmentation-models-pytorch ≥ 0.3, einops, albumentations, opencv-python.

### 2. Prepare Dataset

Download the competition dataset from [Tianchi](https://tianchi.aliyun.com/competition/entrance/532062) and extract:

```bash
cd data/
unzip train.zip    # → data/train/image/*.png + data/train/mask/*.png
unzip test.zip     # → data/test/image/*.png
```

See [`data/README.md`](data/README.md) for detailed format specifications.

### 3. Download Pretrained Backbones

All pretrained weights are loaded relative to the working directory (`code/`). Create a `pretrained/` directory **inside `code/`**:

```bash
mkdir -p code/pretrained/
```

| Backbone | Weight File | Download |
|----------|------------|----------|
| PVT-V2-B4 | `pvt_v2_b4.pth` | [GitHub Releases](https://github.com/whai362/PVT/releases) |
| Swin-Small | `swin_small_patch4_window7_224_22k.pth` | [GitHub](https://github.com/microsoft/Swin-Transformer#getting-started) |
| MIT-B3 | `mit_b3.pth` | [Google Drive](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOGSMikGAiXEa) |
| CoAt-Parallel-Small | `coat_small_7479cf9b.pth` | [GitHub](https://github.com/mlpc-ucsd/CoaT) |
| DualViT-B | *(uses `pvt_v2_b4.pth` for init)* | Same as PVT-V2-B4 |
| ConvNeXt-Large | *(auto-downloaded by timm)* | — |

```bash
# Place weights in code/pretrained/
code/pretrained/
├── pvt_v2_b4.pth                      # Used by: pvt-unet, pvt-unet-mp, dual-daformer
├── swin_small_patch4_window7_224_22k.pth  # Used by: SwinV1-Small
├── mit_b3.pth                         # Used by: Segformer-b3
└── coat_small_7479cf9b.pth            # Used by: coat-daformer
```

> **Note**: DualViT-B uses PVT-V2-B4 weights for initialization (transfer learning from a similar architecture). ConvNeXt-Large is downloaded automatically by `timm` on first use.

### 4. Train

```bash
cd code/

# ★ Recommended: train the best model
python run_train_pvtunet_mp.py

# Other model variants:
python run_train_pvtunet.py            # PVT-V2-B4 + U-Net
python run_train_coat.py               # CoAt-Parallel-Small + DAFormer
python run_train_segformer.py          # SegFormer MIT-B3 + DAFormer
python run_train_convnext_smpunet.py   # ConvNeXt-Large + SMP U-Net
python run_train_dualvit_daformer.py   # DualViT-B + DAFormer
python run_train.py                    # Swin-Small + UPerNet
```

**Output**: `result_teeth-seg/<model>/fold-<N>-tianchi-fold/checkpoint/<epoch>_<model>.pth`

To switch GPU devices, edit `gpu_id` in `Config.py` or set `CUDA_VISIBLE_DEVICES` before running.

### 5. SWA (Stochastic Weight Averaging)

Edit the checkpoint filenames and result directory in `swa.py` to match your training output, then:

```bash
python swa.py
```

**Output**: `result/<model>/fold-<N>-tianchi-fold/<model>-fold-<N>-swa.pth`

### 6. Inference & Submission

Edit the `model` list in `submit.py` to select which SWA checkpoints to ensemble. Uncomment the desired model blocks:

```bash
python submit.py
```

**Output**: `output/<experiment-name>/` — binary mask PNGs (0 or 255), 320×640 each.

## Configuration Reference

All hyperparameters are centralized in [`code/Config.py`](code/Config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `pvt-unet-mp` | Model selector (see architecture table) |
| `dataset` | `Teeth-Seg` | Dataset identifier |
| `seed` | 42 | Global random seed |
| `num_fold` | 5 | Number of cross-validation folds |
| `epochs` | 84 | Total training epochs |
| `start_lr` | 1e-4 | Initial learning rate |
| `min_lr` | 1e-6 | Minimum learning rate (after decay) |
| `train_batch_size` | 8 | Training batch size |
| `val_batch_size` | 16 | Validation batch size (2× train) |
| `num_workers` | 8 | DataLoader worker processes |
| `Dice_loss_weight` | 0.7 | Weight for Dice loss (all models) |
| `Focal_loss_weight` | 0.3 | Weight for Focal loss (pvt-unet, pvt-unet-mp) |
| `BCE_loss_weight` | 0.3 | Weight for BCE loss (Swin, SegFormer, CoAt, ConvNeXt) |
| `TTA` | `True` | Enable 4× rotational test-time augmentation |
| `scale` | 1.0 | Image resize scale factor (applied after padding) |
| `in_chans` | 3 | Input image channels (BGR) |
| `gpu_id` | `'0'` | CUDA device ID(s), e.g. `'0'` or `'0,1'` |
| `drop_rate` | 0.4 | Dropout rate (scheduled during training) |
| `save_root_dir` | `./result` | Root directory for training outputs |

## References

1. Wang et al., "PVT v2: Improved Baselines with Pyramid Vision Transformer", *CVM*, 2022. [[paper]](https://arxiv.org/abs/2106.13797) [[code]](https://github.com/whai362/PVT)
2. Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", *ICCV*, 2021. [[paper]](https://arxiv.org/abs/2103.14030) [[code]](https://github.com/microsoft/Swin-Transformer)
3. Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers", *NeurIPS*, 2021. [[paper]](https://arxiv.org/abs/2105.15203) [[code]](https://github.com/NVlabs/SegFormer)
4. Hoyer et al., "DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation", *CVPR*, 2022. [[paper]](https://arxiv.org/abs/2111.14887) [[code]](https://github.com/lhoyer/DAFormer)
5. Dai et al., "CoaT: Co-Scale Conv-Attentional Image Transformers", *ICCV*, 2021. [[paper]](https://arxiv.org/abs/2104.06399) [[code]](https://github.com/mlpc-ucsd/CoaT)
6. Ren et al., "Dual Vision Transformer", *arXiv*, 2022. [[paper]](https://arxiv.org/abs/2207.04976) [[code]](https://github.com/rentainhe/Dual-ViT)
7. Xiao et al., "Unified Perceptual Parsing for Scene Understanding", *ECCV*, 2018. [[paper]](https://arxiv.org/abs/1807.10221)
8. Lin et al., "Focal Loss for Dense Object Detection", *ICCV*, 2017. [[paper]](https://arxiv.org/abs/1708.02002)
9. Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back", *NeurIPS*, 2019. [[paper]](https://arxiv.org/abs/1907.08610)

## Acknowledgements

- [Tianchi](https://tianchi.aliyun.com/) for hosting the competition and providing the dataset
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) — U-Net decoder implementation
- [timm](https://github.com/huggingface/pytorch-image-models) — Image model library (ConvNeXt, ResNet, layer utilities)
- [albumentations](https://github.com/albumentations-team/albumentations) — Image augmentation library
- [einops](https://github.com/arogozhnikov/einops) — Tensor operations for MeanPool aggregation

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

This repository includes third-party code and references pretrained model weights that are subject to their own licenses. **Please read [NOTICE](NOTICE) carefully before use.** A summary of key license considerations:

### Third-Party Code Licenses

| Component | File(s) | License |
|-----------|---------|---------|
| Google Surface Distance | `Libs/metrics.py`, `Libs/lookup_tables.py` | Apache 2.0 |
| PVT-V2 | `Model/pvt_v2.py`, `Model/pvt_v2_2.py` | Apache 2.0 |
| DAFormer Decoder | `Model/daformer.py` | Apache 2.0 |
| DualViT | `Model/dualvit.py` | Apache 2.0 |
| Swin Transformer V1 | `Model/swin_v1_variable.py` | MIT |
| CoaT | `Model/coat.py` | MIT |
| UPerNet | `Model/upernet.py` | BSD 3-Clause |
| SMP U-Net Decoder | `Model/smp_unet.py` | MIT |
| Lookahead Optimizer | `MyOptimizer/optim.py` | MIT |
| **Mix Transformer (SegFormer)** | **`Model/mit.py`** | **⚠️ NVIDIA SCL (Non-Commercial Only)** |

### Pretrained Model Weights

Pretrained weights are **not included** in this repository and must be downloaded separately. Their license status:

| Weight | License | Commercial Use |
|--------|---------|---------------|
| `pvt_v2_b4.pth` | Apache 2.0 | ✅ Yes |
| `coat_small_7479cf9b.pth` | MIT | ✅ Yes |
| `swin_small_patch4_window7_224_22k.pth` | ImageNet-22k restrictions | ⚠️ Research only |
| `mit_b3.pth` | NVIDIA SCL | ⚠️ Non-commercial only |
| `convnext_large_384_in22ft1k` (via timm) | CC-BY-NC (ImageNet-22k fine-tuned) | ⚠️ Non-commercial only |

> **⚠️ Important:** The `Segformer-b3` model uses the NVIDIA-licensed Mix Transformer encoder, which is restricted to **non-commercial use**. Additionally, several pretrained backbones were trained on **ImageNet-22k**, which was released for **non-commercial research purposes only**. For commercial deployment, replace these components with commercially-licensed alternatives or retrain from scratch.
>
> See [NOTICE](NOTICE) for full license texts and attribution details.

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full details.

```
MIT License

Copyright (c) 2023-2026 terry

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

**Note:** This repository contains code adapted from various open-source projects. Please review the [NOTICE](NOTICE) file for complete attribution and third-party license information.
