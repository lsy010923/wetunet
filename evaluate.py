import argparse
import logging
import time
from pathlib import Path
import os
import sys
import inspect
from typing import Optional, Sequence, Dict, Any, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data_loading import BasicDataset, CarvanaDataset

# Default dataset paths, relative to the repository root. Override them from
# the CLI with --img-dir / --mask-dir if your directory layout differs.
dir_img_val = Path('data/imgs/val')
dir_img_test = Path('data/imgs/test')
dir_mask_val = Path('data/masks/val')
dir_mask_test = Path('data/masks/test')


def build_dataset(img_dir, mask_dir, img_scale):
    try:
        return CarvanaDataset(img_dir, mask_dir, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        return BasicDataset(img_dir, mask_dir, img_scale)


# =========================
# ✅ FIX 1: robust collate for variable-size + non-RGB images + weird masks
# - pads each batch to same H/W (optionally to multiple of 32)
# - forces image to 3-channel CHW
# - forces mask to 2D (H,W) per sample, batched as (B,H,W)
# =========================

def _ensure_chw(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is CHW.
    Accepts:
      - HW
      - CHW
      - HWC (heuristic)
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    if x.dim() == 2:
        # HW -> 1HW
        return x.unsqueeze(0)

    if x.dim() == 3:
        c0, h0, w0 = x.shape
        # Heuristic: if last dim looks like channels (<=4), treat as HWC
        if x.shape[-1] <= 4 and (c0 > 4 or h0 <= 4):
            # HWC -> CHW
            return x.permute(2, 0, 1).contiguous()
        return x  # assume CHW

    raise ValueError(f"Unsupported tensor dims: {x.dim()} shape={tuple(x.shape)}")


def _force_image_3ch(img_chw: torch.Tensor) -> torch.Tensor:
    """
    Make image CHW with C=3.
    """
    img_chw = _ensure_chw(img_chw)

    c, h, w = img_chw.shape
    if c == 3:
        return img_chw
    if c == 1:
        return img_chw.repeat(3, 1, 1)
    if c == 2:
        # uncommon: take first channel and repeat
        return img_chw[:1].repeat(3, 1, 1)
    if c > 3:
        # drop alpha/extra channels
        return img_chw[:3]
    # fallback
    return img_chw[:1].repeat(3, 1, 1)


def _force_mask_hw(mask_any: torch.Tensor) -> torch.Tensor:
    """
    Make mask as HW (2D) int64.
    Handles:
      - HW
      - 1HW
      - CHW (C>1): take first channel
      - HWC (heuristic): permute then take first
    """
    m = mask_any
    if not torch.is_tensor(m):
        m = torch.as_tensor(m)

    if m.dim() == 2:
        return m.to(torch.int64)

    if m.dim() == 3:
        m = _ensure_chw(m)  # now CHW
        if m.shape[0] > 1:
            m = m[:1]
        m = m.squeeze(0)
        return m.to(torch.int64)

    if m.dim() == 4:
        # If someone already returns 1xHxW inside sample (rare), squeeze batch-like dim
        # Keep safest behavior: take first slice, then process
        m = m[0]
        return _force_mask_hw(m)

    raise ValueError(f"Unsupported mask dims: {m.dim()} shape={tuple(m.shape)}")


def _pad_chw(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Pad CHW tensor to (C, target_h, target_w) with zeros.
    """
    c, h, w = x.shape
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    # F.pad pad order: (left, right, top, bottom) for HW, but for CHW it's still last 2 dims
    # so we pad W then H: (left, right, top, bottom)
    return F.pad(x, (0, pad_w, 0, pad_h), value=0)


def _pad_hw(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Pad HW tensor to (target_h, target_w) with zeros.
    """
    h, w = x.shape
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    return F.pad(x, (0, pad_w, 0, pad_h), value=0)


def pad_collate(batch: List[Dict[str, Any]], pad_to_multiple: int = 32) -> Dict[str, torch.Tensor]:
    """
    Collate_fn that pads variable-size samples in a batch to same H/W.
    Outputs:
      image: FloatTensor [B,3,H,W]
      mask : LongTensor  [B,H,W]
    """
    images: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []

    max_h = 0
    max_w = 0

    # 1) normalize shapes, compute max H/W
    for sample in batch:
        img = _force_image_3ch(sample["image"])
        msk = _force_mask_hw(sample["mask"])

        # ensure CHW for image
        img = img.to(torch.float32)
        msk = msk.to(torch.int64)

        _, h, w = img.shape
        max_h = max(max_h, int(h))
        max_w = max(max_w, int(w))

        images.append(img)
        masks.append(msk)

    # 2) optional: pad to multiple of N (good for encoder stride / speed)
    if pad_to_multiple and pad_to_multiple > 1:
        mh = int((max_h + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        mw = int((max_w + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        max_h, max_w = mh, mw

    # 3) pad each sample to same size
    padded_images = [_pad_chw(im, max_h, max_w) for im in images]
    padded_masks = [_pad_hw(msk, max_h, max_w) for msk in masks]

    return {
        "image": torch.stack(padded_images, dim=0),
        "mask": torch.stack(padded_masks, dim=0),
    }


def extract_state_dict(ckpt: Any) -> Any:
    """
    Compatible with:
      - raw state_dict (dict of tensors)
      - wrapped ckpt: {'state_dict': ...} / {'model_state_dict': ...} / {'model': ...} / {'net': ...}
    """
    if isinstance(ckpt, dict):
        for k in ['state_dict', 'model_state_dict', 'model', 'net']:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # sometimes ckpt itself IS state_dict
        if all(isinstance(x, str) for x in ckpt.keys()):
            return ckpt
    return ckpt


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove 'module.' prefix from DDP/DataParallel checkpoints."""
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith('module.') for k in state_dict.keys()):
        return {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    return state_dict


def drop_junk_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove non-weight keys that sometimes get mixed into state_dict.
    Your current bug: 'mask_values' is inside state_dict and breaks strict load.
    """
    if not isinstance(state_dict, dict):
        return state_dict
    state_dict.pop('mask_values', None)  # ✅ the key that is breaking your load
    return state_dict


def is_wetbird_state_dict(state_dict: Dict[str, Any]) -> bool:
    """
    WetBirdNet checkpoints always carry MobileNetV3 backbone keys
    ('backbone.features.*'). Used to fail early with a clear message when a
    checkpoint from an unrelated architecture is passed in.
    """
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return False
    return any(k.startswith("backbone.features.") for k in state_dict.keys())


def infer_n_classes_from_state_dict(state_dict: Dict[str, Any], default: int = 2) -> int:
    """
    WetBirdNet head: outc.weight shape = [n_classes, 64, 1, 1]
    """
    if not isinstance(state_dict, dict):
        return default

    for k in ["outc.weight", "outc.conv.weight"]:
        if k in state_dict and isinstance(state_dict[k], torch.Tensor):
            w = state_dict[k]
            if w.dim() >= 1:
                return int(w.shape[0])

    # fallback
    return default


def filter_init_kwargs(cls, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Sequence[str]]:
    """
    Keep only kwargs accepted by cls.__init__ signature.
    Returns (filtered_kwargs, accepted_keys)
    """
    sig = inspect.signature(cls.__init__)
    accepted = []
    out = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            out[k] = v
            accepted.append(k)
    return out, accepted


@torch.no_grad()
def evaluate(
    net,
    dataloader,
    device,
    amp: bool = False,
    num_classes: Optional[int] = None,
    return_all: bool = False,
    measure_speed: bool = False,
    threshold: float = 0.5,
    thresholds: Optional[Sequence[float]] = None,
    objective: str = "dice",  # "dice" | "iou_fg" | "miou"
    ignore_index: Optional[int] = None,
):
    """
    Binary (single-channel logits) critical behavior:
      - Always evaluates with 2 classes (bg=0, fg=1) internally.
      - If thresholds is provided, scans thresholds and picks the best by `objective`.

    objective:
      - "dice"   : maximize foreground Dice
      - "iou_fg" : maximize foreground IoU
      - "miou"   : maximize mean IoU (bg+fg)/2 for binary
    """
    net.eval()

    # Speed timing
    start_time = None
    total_images = 0
    if measure_speed:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

    iterator = iter(dataloader)
    try:
        first_batch = next(iterator)
    except StopIteration:
        if not return_all:
            return 0.0
        return {
            "Score": 0.0,
            "BestThreshold": None,
            "Dice_fg": 0.0,
            "IoU_fg": 0.0,
            "mIoU": 0.0,
            "mPA": 0.0,
            "Accuracy": 0.0,
            "Recall_fg": 0.0,
            "Precision_fg": 0.0,
            "F1_fg": 0.0,
            "FPS": None,
        }

    def _to_device(batch):
        images = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        true_masks = batch["mask"].to(device=device)
        return images, true_masks

    images0, true0 = _to_device(first_batch)
    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        logits0 = net(images0)

    if logits0.dim() != 4:
        raise ValueError(f"Expected logits [B,C,H,W], got {tuple(logits0.shape)}")

    is_binary = (logits0.shape[1] == 1)
    if not is_binary:
        C = logits0.shape[1]
        nc = num_classes if num_classes is not None else C
        if nc != C:
            nc = C
    else:
        nc = 2  # ALWAYS 2 for binary evaluation

    # Choose thresholds list
    if is_binary and thresholds is None:
        thresholds = [float(threshold)]
    if not is_binary:
        thresholds = None

    # Binary helpers
    def _init_counts(n_thr: int):
        z = torch.zeros((n_thr,), dtype=torch.float64, device=device)
        return z.clone(), z.clone(), z.clone(), z.clone()

    conf_mat = torch.zeros((nc, nc), dtype=torch.long, device=device) if not is_binary else None
    total_correct = 0
    total_pixels = 0
    eps = 1e-7

    if is_binary:
        thr_list = list(thresholds)
        n_thr = len(thr_list)
        TP, FP, FN, TN = _init_counts(n_thr)

        def _accumulate_binary(logits, true_masks_local):
            nonlocal TP, FP, FN, TN, total_correct, total_pixels, total_images
            probs = torch.sigmoid(logits[:, 0])  # [B,H,W]

            # ✅ expects true_masks_local as [B,H,W] (our pad_collate guarantees this)
            true_bin = (true_masks_local > 0).to(torch.bool)

            valid = torch.ones_like(true_bin, dtype=torch.bool)
            if ignore_index is not None:
                valid &= (true_masks_local != ignore_index)

            total_images += int(probs.shape[0])

            for i, t in enumerate(thr_list):
                pred = (probs > t)
                pred = pred & valid
                tru = true_bin & valid

                TP[i] += (pred & tru).sum().to(torch.float64)
                FP[i] += (pred & (~tru)).sum().to(torch.float64)
                FN[i] += ((~pred) & tru).sum().to(torch.float64)
                TN[i] += ((~pred) & (~tru)).sum().to(torch.float64)

            # Accuracy tracked using first threshold
            pred0 = (probs > float(thr_list[0])).to(torch.long)
            true0l = true_bin.to(torch.long)
            total_correct += (pred0[valid] == true0l[valid]).sum().item()
            total_pixels += int(valid.sum().item())

        _accumulate_binary(logits0, true0)
        for batch in iterator:
            images, true_masks = _to_device(batch)
            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                logits = net(images)
            if logits.shape[1] != 1:
                raise RuntimeError("Model output channel changed during evaluation.")
            _accumulate_binary(logits, true_masks)

        dice = (2 * TP) / (2 * TP + FP + FN + eps)
        iou_fg = TP / (TP + FP + FN + eps)
        iou_bg = TN / (TN + FP + FN + eps)
        miou = 0.5 * (iou_fg + iou_bg)

        recall_fg = TP / (TP + FN + eps)
        precision_fg = TP / (TP + FP + eps)
        f1_fg = (2 * precision_fg * recall_fg) / (precision_fg + recall_fg + eps)

        recall_bg = TN / (TN + FP + eps)
        mpa = 0.5 * (recall_bg + recall_fg)

        if objective == "dice":
            score_arr = dice
        elif objective == "iou_fg":
            score_arr = iou_fg
        elif objective == "miou":
            score_arr = miou
        else:
            raise ValueError("objective must be one of: dice | iou_fg | miou")

        best_i = int(torch.argmax(score_arr).item())
        best_thr = float(thr_list[best_i])

        if measure_speed and start_time is not None:
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time = time.time() - start_time
            fps = total_images / max(total_time, 1e-6)
        else:
            fps = None

        metrics = {
            "Score": float(score_arr[best_i].item()),
            "BestThreshold": best_thr,
            "Dice_fg": float(dice[best_i].item()),
            "IoU_fg": float(iou_fg[best_i].item()),
            "mIoU": float(miou[best_i].item()),
            "mPA": float(mpa[best_i].item()),
            "Accuracy": float(total_correct / max(total_pixels, 1)),
            "Recall_fg": float(recall_fg[best_i].item()),
            "Precision_fg": float(precision_fg[best_i].item()),
            "F1_fg": float(f1_fg[best_i].item()),
            "FPS": fps,
            "BinaryOutput": True,
            "Objective": objective,
            "ThresholdsScanned": len(thr_list),
        }
        return metrics if return_all else metrics["Score"]

    # ---------------- multiclass path ----------------
    def _update_conf(preds, true_masks_local):
        nonlocal conf_mat, total_correct, total_pixels, total_images
        true_flat = true_masks_local.view(-1).long()
        pred_flat = preds.view(-1).long()

        valid = torch.ones_like(true_flat, dtype=torch.bool)
        if ignore_index is not None:
            valid &= (true_flat != ignore_index)
        valid &= (true_flat >= 0) & (true_flat < nc)
        valid &= (pred_flat >= 0) & (pred_flat < nc)

        true_v = true_flat[valid]
        pred_v = pred_flat[valid]

        total_images += int(preds.shape[0])
        total_pixels += int(true_v.numel())
        total_correct += (true_v == pred_v).sum().item()

        k = (true_v * nc + pred_v).to(torch.int64)
        conf_mat += torch.bincount(k, minlength=nc * nc).reshape(nc, nc)

    probs0 = F.softmax(logits0, dim=1)
    preds0 = probs0.argmax(dim=1)
    _update_conf(preds0, true0.long())

    for batch in iterator:
        images, true_masks = _to_device(batch)
        with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
            logits = net(images)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        _update_conf(preds, true_masks.long())

    if measure_speed and start_time is not None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time = time.time() - start_time
        fps = total_images / max(total_time, 1e-6)
    else:
        fps = None

    conf = conf_mat.detach().cpu().float()
    tp = torch.diag(conf)
    per_class_total = conf.sum(dim=1)
    per_class_pred = conf.sum(dim=0)

    pa_per_class = tp / (per_class_total + eps)
    mPA = pa_per_class.mean().item()

    iou_per_class = tp / (per_class_total + per_class_pred - tp + eps)
    mIoU = iou_per_class.mean().item()

    accuracy = total_correct / max(total_pixels, 1)

    fg_idx = 1 if nc > 1 else 0
    TP_fg = tp[fg_idx].item()
    FN_fg = (per_class_total[fg_idx] - tp[fg_idx]).item()
    FP_fg = (per_class_pred[fg_idx] - tp[fg_idx]).item()

    dice_fg = 2 * TP_fg / (2 * TP_fg + FP_fg + FN_fg + eps)
    recall_fg = TP_fg / (TP_fg + FN_fg + eps)
    precision_fg = TP_fg / (TP_fg + FP_fg + eps)
    f1_fg = 2 * precision_fg * recall_fg / (precision_fg + recall_fg + eps)
    iou_fg = TP_fg / (TP_fg + FP_fg + FN_fg + eps)

    score = dice_fg if objective == "dice" else (mIoU if objective == "miou" else iou_fg)

    metrics = {
        "Score": float(score),
        "BestThreshold": None,
        "Dice_fg": float(dice_fg),
        "IoU_fg": float(iou_fg),
        "mIoU": float(mIoU),
        "mPA": float(mPA),
        "Accuracy": float(accuracy),
        "Recall_fg": float(recall_fg),
        "Precision_fg": float(precision_fg),
        "F1_fg": float(f1_fg),
        "FPS": fps,
        "BinaryOutput": False,
        "Objective": objective,
        "ThresholdsScanned": 0,
    }
    return metrics if return_all else metrics["Score"]


@torch.no_grad()
def compute_flops_params(net, loader, device, logger):
    params = sum(p.numel() for p in net.parameters())
    params_m = params / 1e6

    # ✅ With pad_collate, this will never crash on variable-size images
    sample = next(iter(loader))
    dummy = sample['image'][:1].to(device=device, dtype=torch.float32).contiguous(memory_format=torch.channels_last)

    flops_g = None
    try:
        from thop import profile
        flops, _ = profile(net, inputs=(dummy,), verbose=False)
        flops_g = flops / 1e9
        logger.info(f'FLOPs computed by thop on input shape {tuple(dummy.shape)}')
    except Exception as e:
        logger.warning(f'FLOPs skipped (thop not available or failed): {e}')
        logger.warning('If you want FLOPs, run: pip install thop')

    result = {'Params(M)': params_m, 'Params': params}
    if flops_g is not None:
        result['FLOPs(G)'] = flops_g
    return result


def get_eval_args():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model with metrics + optional threshold scan')
    parser.add_argument('--checkpoint', '-m', type=str, required=True, help='Path to the .pth checkpoint file')
    parser.add_argument('--batch-size', '-b', type=int, default=8)
    parser.add_argument('--scale', '-s', type=float, default=1.0)
    parser.add_argument('--classes', '-c', type=int, default=1, help='Overridden if inferred from the checkpoint head')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--img-dir', type=str, default=None,
                        help='Override image directory for evaluation')
    parser.add_argument('--mask-dir', type=str, default=None,
                        help='Override mask directory for evaluation')
    parser.add_argument('--amp', action='store_true', default=False)

    parser.add_argument('--threshold', type=float, default=0.5, help='Used if not scanning thresholds')
    parser.add_argument('--thr-scan', action='store_true', default=False, help='Enable threshold scanning (binary only)')
    parser.add_argument('--thr-min', type=float, default=0.30)
    parser.add_argument('--thr-max', type=float, default=0.70)
    parser.add_argument('--thr-step', type=float, default=0.05)
    parser.add_argument('--objective', type=str, default='dice', choices=['dice', 'iou_fg', 'miou'])
    parser.add_argument('--ignore-index', type=int, default=None)

    # WetBirdNet optional init params
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--normalize-input', action='store_true', default=False)
    parser.add_argument('--drop-path', type=float, default=0.0)
    parser.add_argument('--scse-r', type=int, default=8)

    # load strictness
    parser.add_argument('--strict', action='store_true', default=True, help='Use strict=True for load_state_dict')
    parser.add_argument('--no-strict', action='store_true', default=False, help='Set strict=False for load_state_dict')

    return parser.parse_args()


def torch_load_safe(path: str, map_location: str = "cpu"):
    """
    Prefer weights_only=True when available (newer torch), fallback otherwise.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


if __name__ == '__main__':
    args = get_eval_args()

    log_dir = Path('./logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'eval_seg.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')

    # Load checkpoint
    ckpt = torch_load_safe(args.checkpoint, map_location='cpu')
    state_dict = extract_state_dict(ckpt)

    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint does not contain a valid state_dict dict.")

    # Clean prefixes + junk keys (✅ this fixes your 'mask_values' crash)
    state_dict = strip_module_prefix(state_dict)
    state_dict = drop_junk_keys(state_dict)

    # Sanity check: this release only ships WetBirdNet
    if not is_wetbird_state_dict(state_dict):
        raise RuntimeError(
            "Checkpoint does not look like a WetBirdNet checkpoint "
            "(no 'backbone.features.*' keys). Only WetBirdNet is supported in this release."
        )

    # Infer number of output classes from head
    inferred_classes = infer_n_classes_from_state_dict(state_dict, default=args.classes)
    if inferred_classes != args.classes:
        logger.info(f'Overriding --classes {args.classes} -> {inferred_classes} (inferred from checkpoint head)')
    args.classes = inferred_classes

    # Build model
    from wetland_unet import WetBirdNet

    init_kwargs = {
        "n_classes": args.classes,
        "pretrained": bool(args.pretrained),
        "normalize_input": bool(args.normalize_input),
        "drop_path": float(args.drop_path),
        "scse_r": int(args.scse_r),
    }
    filtered_kwargs, accepted = filter_init_kwargs(WetBirdNet, init_kwargs)
    logger.info(f'WetBirdNet init accepted keys: {sorted(list(accepted))}')

    net = WetBirdNet(**filtered_kwargs)

    # Load weights
    strict = True
    if args.no_strict:
        strict = False
    elif args.strict:
        strict = True

    net.load_state_dict(state_dict, strict=strict)
    net.to(device=device)
    logger.info(f'Loaded model from {args.checkpoint} (strict={strict})')

    # Dataset/loader
    if args.img_dir is not None or args.mask_dir is not None:
        if args.img_dir is None or args.mask_dir is None:
            raise ValueError("--img-dir and --mask-dir must be provided together")
        img_dir, mask_dir = Path(args.img_dir), Path(args.mask_dir)
    elif args.split == 'val':
        img_dir, mask_dir = dir_img_val, dir_mask_val
    else:
        img_dir, mask_dir = dir_img_test, dir_mask_test

    dataset = build_dataset(img_dir, mask_dir, args.scale)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
        collate_fn=pad_collate,  # ✅ FIX: prevents stack crash on variable-size images
    )
    logger.info(f'Evaluating split={args.split}, size={len(dataset)}')

    fp = compute_flops_params(net, loader, device, logger)

    thr_list = None
    if args.thr_scan:
        t = args.thr_min
        thr_list = []
        while t <= args.thr_max + 1e-9:
            thr_list.append(round(float(t), 6))
            t += args.thr_step

    metrics = evaluate(
        net=net,
        dataloader=loader,
        device=device,
        amp=args.amp,
        num_classes=args.classes,
        return_all=True,
        measure_speed=True,
        threshold=args.threshold,
        thresholds=thr_list,
        objective=args.objective,
        ignore_index=args.ignore_index,
    )
    metrics.update(fp)

    logger.info('====== Evaluation Results ======')
    for k, v in metrics.items():
        if v is None:
            logger.info(f'{k}: N/A')
        else:
            if k in ['FPS', 'Params(M)', 'FLOPs(G)'] and isinstance(v, (int, float)):
                logger.info(f'{k}: {v:.3f}')
            elif k in ['Params']:
                logger.info(f'{k}: {int(v)}')
            elif isinstance(v, (int, float)):
                logger.info(f'{k}: {v:.4f}')
            else:
                logger.info(f'{k}: {v}')
