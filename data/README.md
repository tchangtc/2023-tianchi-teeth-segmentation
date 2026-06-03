# Dataset — 2D Teeth Segmentation

## Overview

The dataset consists of dental panoramic X-ray images with pixel-level binary segmentation masks.

- **Training set**: 2000 images (with masks)
- **Test set**: 500 images (without masks)

## Download

Download from the Tianchi competition page:
- [2023 天池牙齿分割大赛](https://tianchi.aliyun.com/competition/entrance/532062)

## Setup

After downloading, place and unzip the archives as follows:

```
tianchi-2023-teeth-segmentation/
├── code/               # source code
│   ├── train.csv       # data index (relative paths point here ↓)
│   ├── test.csv
│   └── ...
└── data/
    ├── train/
    │   ├── image/      # 2000 PNG images (320×640, 3ch BGR)
    │   │   ├── A-1.png
    │   │   ├── A-10.png
    │   │   └── ...
    │   └── mask/       # 2000 binary mask PNGs (320×640, 1ch)
    │       ├── A-1.png
    │       └── ...
    └── test/
        └── image/      # 500 PNG images (320×640, 3ch BGR)
            ├── 1.png
            └── ...
```

```bash
# From the project root:
cd data/
unzip train.zip   # should create data/train/image/ and data/train/mask/
unzip test.zip    # should create data/test/image/
```

> **Note**: The `train.csv` and `test.csv` files use relative paths (`../data/train/...`, `../data/test/...`)
> so they work when running scripts from the `code/` directory.

## Data Format

| Item | Shape | Channels | Dtype | Range |
|------|-------|----------|-------|-------|
| Image | 320 × 640 | 3 (BGR) | uint8 | 0–255 |
| Mask | 320 × 640 | 1 | uint8 | 0 (background) / 255 (teeth) |

## Preprocessing (in code)

1. **Padding**: images are padded to 640×640 (centered)
2. **Optional scaling**: `CFG.scale` (default 1.0)
3. **Normalization**: pixel values / 255.0
4. **Mask binarization**: `mask[mask > 0] = 1`
