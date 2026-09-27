# WetBirdNet

PyTorch implementation of **WetBirdNet**, a U-Net style segmentation network for
binary **wetland waterbird segmentation** from high-resolution imagery.

> **Dataset / 数据集**: The dataset is not publicly distributed. To request
> access, please contact **1294239487@qq.com**.
> （数据集不随代码公开，如需获取请发送邮件联系。）

The codebase is a trimmed, self-contained release: only the model, the training
and evaluation pipelines, and an inference script are included. All experiment
scaffolding, ablations, logs and datasets from the development repository have
been removed.

## Architecture

WetBirdNet follows the U-Net topology with a MobileNetV3-Large encoder and three
custom components:

| Component | Role |
| --- | --- |
| **GSCAM** (Global Spatial-Channel Attention Module) | Replaces every SqueezeExcitation block inside the MobileNetV3-Large backbone with a dual-branch channel + spatial attention gate. |
| **MS-ADCAM** (Multi-Scale Attention Dual-Channel Attention Module) | Placed at the bottleneck to re-weight multi-scale context before decoding. |
| **HFRM** (Hybrid Feature Refinement Module) | Each decoder stage: adaptive skip fusion + large-kernel refinement + optional channel-spatial gate. |

Encoder features are tapped at indices 1, 3, 6, 12 and 16, with channel widths
16, 24, 40, 112 and 960. The decoder upsamples four times, projects every skip
connection to 1x1 convolutions, and finally upsamples back to the input
resolution with a 1x1 classification head.

Training uses a weighted **0.3 BCE + 0.7 Dice** loss with optional pos_weight,
automatic mixed precision, gradient clipping, and a ReduceLROnPlateau scheduler
driven by a **fixed-threshold** validation score (the optional threshold scan is
reported for inspection only, never used to step the scheduler).

## Repository structure

```
WetBirdNet/
├── wetland_unet.py        # WetBirdNet model (+ GSCAM / MS-ADCAM / HFRM blocks)
├── train.py               # Training entry point
├── evaluate.py            # Evaluation + metrics (Dice / IoU / mIoU / FPS)
├── predict.py             # Inference: images -> binary masks
├── utils/
│   └── data_loading.py    # BasicDataset / CarvanaDataset
├── requirements.txt
├── Dockerfile
└── data/                  # Place your dataset here (not tracked by git)
```

## Installation

Python 3.9+ and PyTorch 2.3+ are required.

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended. For a containerized environment:

```bash
docker build -t wetbirdnet .
docker run --rm --gpus all --shm-size=8g -it -v $PWD:/workspace/wetbirdnet wetbirdnet
```

## Data layout

Images and masks are matched by **filename stem**, so `data/imgs/train/001.jpg`
pairs with `data/masks/train/001.png`. Masks must be single-channel; every
non-zero pixel is treated as foreground (bird).

```
data/
├── imgs/
│   ├── train/
│   ├── val/
│   └── test/
└── masks/
    ├── train/
    ├── val/
    └── test/
```

The dataset used in the paper is not redistributed with this repository — see
the dataset note at the top of this README to request access. Any binary
segmentation dataset with the layout above works out of the box.

## Training

```bash
python train.py \
    --train-img data/imgs/train --train-mask data/masks/train \
    --val-img   data/imgs/val   --val-mask   data/masks/val \
    --epochs 100 --batch-size 8 --amp
```

Useful flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--epochs`, `-e` | `100` | Number of epochs |
| `--batch-size`, `-b` | `8` | Batch size |
| `--learning-rate`, `-l` | `1e-4` | AdamW learning rate |
| `--scale`, `-s` | `1.0` | Input downscaling factor |
| `--w-bce` / `--w-dice` | `0.3` / `0.7` | Loss weights |
| `--pos-weight` | `1.0` | BCE positive class weight (`<=0` disables it) |
| `--val-objective` | `dice` | `dice` / `iou_fg` / `miou` |
| `--seed` | `None` | Seed for reproducible runs |
| `--resume` | `None` | Resume from a checkpoint |

Outputs:

```
checkpoints/wetbird_net/
├── checkpoint_best.pth     # best fixed-threshold validation score
├── checkpoint_last.pth
├── checkpoint_epochN.pth   # every --checkpoint-interval epochs
├── metrics.csv             # per-epoch metrics
└── logs/                   # per-run text logs
```

Every checkpoint stores the model / optimizer / scheduler / AMP scaler state plus
the arguments and validation metrics of the epoch that produced it, and the
`mask_values` list needed to map predictions back to original label values.

## Evaluation

```bash
python evaluate.py \
    --checkpoint checkpoints/wetbird_net/checkpoint_best.pth \
    --img-dir data/imgs/test --mask-dir data/masks/test \
    --split test --thr-scan
```

`evaluate.py` reports foreground Dice, foreground IoU, mIoU, FPS, parameter count
and (when `thop` is installed) FLOPs. Add `--thr-scan` to sweep thresholds and
report the best one. The number of output classes is inferred from the
checkpoint head, so `--classes` normally does not need to be set.

## Inference

```bash
# A whole directory
python predict.py --model checkpoints/wetbird_net/checkpoint_best.pth \
                  --input data/imgs/test --output predictions

# A single image, plus a red overlay for visual inspection
python predict.py --model checkpoints/wetbird_net/checkpoint_best.pth \
                  --input data/imgs/test/001.jpg --output predictions --save-overlay
```

Masks are written as 8-bit PNGs (0 = background, 255 = bird) at the original
image resolution. Use `--threshold` to tune the foreground cut-off and
`--no-cuda` to force CPU inference.

## License and acknowledgements

Released under the **GNU General Public License v3.0** (see `LICENSE`), inherited
from [milesial/PyTorch-UNet](https://github.com/milesial/Pytorch-UNet), which
this project is derived from. The U-Net architecture originates from
[Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597).

If you use this code, please cite the corresponding paper.
