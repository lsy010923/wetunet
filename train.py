"""Training entry point for WetBirdNet.

WetBirdNet: a U-Net style architecture for binary wetland waterbird
segmentation. Encoder is MobileNetV3-Large with SE blocks replaced by GSCAM,
the bottleneck uses MS-ADCAM, and the decoder uses HFRM refinement.
"""

import argparse
import logging
import sys
import csv
import time
import random
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms.functional as TF

from wetland_unet import WetBirdNet
from utils.data_loading import BasicDataset, CarvanaDataset
from evaluate import evaluate


# Dataset layout expected by default (override with the CLI flags below):
#   data/imgs/train, data/masks/train, data/imgs/val, data/masks/val
DEFAULT_TRAIN_IMG = "data/imgs/train"
DEFAULT_VAL_IMG = "data/imgs/val"
DEFAULT_TRAIN_MSK = "data/masks/train"
DEFAULT_VAL_MSK = "data/masks/val"


class DiceBCELoss(nn.Module):
    """
    Binary segmentation loss:
      inputs:  [B,1,H,W] logits
      targets: [B,1,H,W] float {0,1}

    Computes the loss in float32 for AMP stability and moves pos_weight to the
    input device inside forward() to avoid device mismatch.
    """

    def __init__(self, w_bce: float = 0.3, w_dice: float = 0.7, pos_weight: Optional[float] = None):
        super().__init__()
        self.w_bce = float(w_bce)
        self.w_dice = float(w_dice)

        if pos_weight is None:
            self.register_buffer("pos_weight", None, persistent=False)
        else:
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        inputs_f = inputs.float()
        targets_f = targets.float()

        if self.pos_weight is None:
            bce_loss = F.binary_cross_entropy_with_logits(inputs_f, targets_f, reduction="mean")
        else:
            pw = self.pos_weight.to(device=inputs_f.device, dtype=inputs_f.dtype)
            bce_loss = F.binary_cross_entropy_with_logits(inputs_f, targets_f, pos_weight=pw, reduction="mean")

        probs = torch.sigmoid(inputs_f)
        dims = (1, 2, 3)
        smooth = 1e-5
        inter = (probs * targets_f).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_f.sum(dim=dims)
        dice = (2.0 * inter + smooth) / (union + smooth)
        dice_loss = (1.0 - dice).mean()

        return self.w_bce * bce_loss + self.w_dice * dice_loss


def build_dataset(img_dir: Path, mask_dir: Path, img_scale: float):
    try:
        return CarvanaDataset(img_dir, mask_dir, img_scale)
    except Exception:
        return BasicDataset(img_dir, mask_dir, img_scale)


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for repeatable training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int, base_seed: int) -> None:
    """Keep DataLoader worker randomness tied to the experiment seed."""
    worker_seed = int(base_seed) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_model(n_classes: int) -> torch.nn.Module:
    """Build WetBirdNet, the paper's main model."""
    return WetBirdNet(n_classes=n_classes)


def save_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_checkpoint(path: str, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt
    return {"model_state_dict": ckpt}


def make_threshold_list(
    thr_scan: bool,
    thr_min: float,
    thr_max: float,
    thr_step: float,
    fallback: float
) -> Optional[List[float]]:
    if not thr_scan:
        return None
    t = thr_min
    out: List[float] = []
    while t <= thr_max + 1e-9:
        out.append(round(float(t), 6))
        t += thr_step
    if len(out) == 0:
        return [float(fallback)]
    return out


def setup_logging(log_dir: Path, run_name: str = "train") -> Path:
    """Log to both console and file. Returns the log file path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{run_name}_{ts}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return log_file


def init_metrics_csv(csv_path: Path) -> None:
    """Create metrics.csv with a header if it does not exist yet."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch",
                "train_loss",

                # score used by the scheduler (fixed threshold by default)
                "sched_objective",
                "sched_score",
                "sched_threshold",

                # threshold scan, for observation only
                "scan_enabled",
                "scan_score",
                "best_threshold",

                # remaining metrics
                "dice_fg",
                "iou_fg",
                "miou",

                "lr",
                "best_score_so_far",
                "epoch_time_sec"
            ])


def append_metrics_csv(
    csv_path: Path,
    epoch: int,
    train_loss: float,

    sched_objective: str,
    sched_score: float,
    sched_threshold: float,

    scan_enabled: bool,
    scan_score: float,
    best_threshold: float,

    val_metrics_for_show: Dict[str, Any],
    lr: float,
    best_score: float,
    epoch_time_sec: float
) -> None:
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            int(epoch),
            float(train_loss),

            str(sched_objective),
            float(sched_score),
            float(sched_threshold),

            int(bool(scan_enabled)),
            float(scan_score),
            float(best_threshold),

            float(val_metrics_for_show.get("Dice_fg", 0.0)),
            float(val_metrics_for_show.get("IoU_fg", 0.0)),
            float(val_metrics_for_show.get("mIoU", 0.0)),

            float(lr),
            float(best_score),
            float(epoch_time_sec)
        ])


def train_model(
    model: torch.nn.Module,
    device: torch.device,
    train_img: Path,
    train_mask: Path,
    val_img: Path,
    val_mask: Path,
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    img_scale: float = 1.0,
    amp: bool = False,
    w_bce: float = 0.3,
    w_dice: float = 0.7,
    pos_weight: Optional[float] = None,

    # Objective used by evaluate() and by the reported score
    val_objective: str = "dice",  # dice | iou_fg | miou

    # Threshold scan: reports thr* and the upper bound of the metric. It is not
    # recommended as a scheduler driver because the scanned value is noisier.
    thr_scan: bool = True,
    thr_min: float = 0.30,
    thr_max: float = 0.70,
    thr_step: float = 0.05,
    threshold_fallback: float = 0.5,

    scheduler_metric: str = "fixed",  # fixed | scan

    lr_patience: int = 10,
    lr_factor: float = 0.5,
    lr_threshold: float = 1e-4,
    lr_cooldown: int = 2,
    min_lr: float = 1e-6,

    checkpoint_dir: Path = Path("./checkpoints/wetbird_net"),
    checkpoint_interval: int = 10,
    grad_clip: float = 1.0,
    resume_ckpt: Optional[str] = None,
    metrics_csv: Optional[Path] = None,
    seed: Optional[int] = None,
):
    train_set = build_dataset(train_img, train_mask, img_scale)
    val_set = build_dataset(val_img, val_mask, img_scale)

    n_train = len(train_set)
    n_val = len(val_set)

    num_workers = 1
    loader_generator = None
    worker_init_fn = None
    if seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(seed))
        worker_init_fn = partial(seed_worker, base_seed=int(seed))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=num_workers, pin_memory=True,
        generator=loader_generator, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True,
        generator=loader_generator, worker_init_fn=worker_init_fn
    )

    logging.info(f"""Starting training:
    Model:             WetBirdNet
    Train img:         {train_img}
    Train mask:        {train_mask}
    Val img:           {val_img}
    Val mask:          {val_mask}
    Epochs:            {epochs}
    Batch size:        {batch_size}
    LR:                {learning_rate}
    Weight decay:      {weight_decay}
    Train/Val size:    {n_train}/{n_val}
    Device:            {device.type}
    AMP:               {amp}
    Seed:              {seed}
    Loss:              BCE(w={w_bce}) + Dice(w={w_dice})
    pos_weight:        {pos_weight}
    Val objective:     {val_objective}

    Thr scan:          {thr_scan}  [{thr_min}, {thr_max}, step={thr_step}]  fallback={threshold_fallback}
    Scheduler metric:  {scheduler_metric}  (fixed=use fallback threshold; scan=use thr_scan best)
    Plateau params:    patience={lr_patience} factor={lr_factor} threshold={lr_threshold} cooldown={lr_cooldown} min_lr={min_lr}

    Ckpt dir:          {checkpoint_dir}  interval={checkpoint_interval}
    Metrics CSV:       {metrics_csv}
""")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=int(lr_patience),
        factor=float(lr_factor),
        threshold=float(lr_threshold),
        threshold_mode="abs",
        cooldown=int(lr_cooldown),
        min_lr=float(min_lr),
    )

    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    else:
        scaler = torch.amp.GradScaler(enabled=amp)

    criterion = DiceBCELoss(w_bce=w_bce, w_dice=w_dice, pos_weight=pos_weight)

    start_epoch = 1
    best_score = -1e9

    if resume_ckpt:
        ckpt = load_checkpoint(resume_ckpt, device)
        state = ckpt.get("model_state_dict", ckpt)

        if isinstance(state, dict) and "mask_values" in state:
            del state["mask_values"]

        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        model.load_state_dict(state, strict=True)

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and amp and ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("best_score", best_score))
        logging.info(f"Resumed from {resume_ckpt} (start_epoch={start_epoch}, best_score={best_score:.6f})")

    if metrics_csv is not None:
        init_metrics_csv(metrics_csv)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        model.train()
        running_loss = 0.0

        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                images = batch["image"]
                true_masks = batch["mask"]

                # Simple augmentation: horizontal / vertical flips
                if torch.rand(1).item() < 0.5:
                    images = TF.hflip(images)
                    true_masks = TF.hflip(true_masks)
                if torch.rand(1).item() < 0.5:
                    images = TF.vflip(images)
                    true_masks = TF.vflip(true_masks)

                true_masks = (true_masks > 0).float()

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                    logits = model(images)
                    loss = criterion(logits, true_masks)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

                running_loss += float(loss.item())
                pbar.update(images.shape[0])
                pbar.set_postfix(loss=float(loss.item()))

        train_loss_avg = running_loss / max(len(train_loader), 1)

        # Fixed-threshold validation: stable, used to drive the scheduler and to
        # select the best checkpoint.
        val_metrics_fixed = evaluate(
            net=model,
            dataloader=val_loader,
            device=device,
            amp=amp,
            return_all=True,
            measure_speed=False,
            threshold=threshold_fallback,
            thresholds=None,  # no scan
            objective=val_objective,
            ignore_index=None,
        )
        fixed_score = float(val_metrics_fixed["Score"])

        # Optional threshold scan, for thr* and metric upper bound only
        scan_score = fixed_score
        best_thr = float(val_metrics_fixed.get("BestThreshold", threshold_fallback))
        val_metrics_show = val_metrics_fixed

        if thr_scan:
            thr_list = make_threshold_list(thr_scan, thr_min, thr_max, thr_step, threshold_fallback)
            val_metrics_scan = evaluate(
                net=model,
                dataloader=val_loader,
                device=device,
                amp=amp,
                return_all=True,
                measure_speed=False,
                threshold=threshold_fallback,
                thresholds=thr_list,
                objective=val_objective,
                ignore_index=None,
            )
            scan_score = float(val_metrics_scan["Score"])
            best_thr = float(val_metrics_scan.get("BestThreshold", threshold_fallback))
            val_metrics_show = val_metrics_scan

        if scheduler_metric == "scan" and thr_scan:
            sched_score = scan_score
        else:
            sched_score = fixed_score

        scheduler.step(sched_score)

        lr_now = optimizer.param_groups[0]["lr"]

        logging.info(
            f"Epoch {epoch:03d} | train_loss={train_loss_avg:.6f} | "
            f"Score_fixed({val_objective})={fixed_score:.6f} | "
            f"Score_scan({val_objective})={scan_score:.6f} | "
            f"Dice_fg={val_metrics_show['Dice_fg']:.6f} | IoU_fg={val_metrics_show['IoU_fg']:.6f} | mIoU={val_metrics_show['mIoU']:.6f} | "
            f"thr*={best_thr:.2f} | lr={lr_now:.2e}"
        )

        is_best = fixed_score > best_score
        if is_best:
            best_score = fixed_score
            logging.info(f"New BEST Score_fixed({val_objective})={best_score:.6f}")

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "epoch": epoch,
            "best_score": best_score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if amp else None,

            # both validation results are stored for later inspection
            "val_metrics_fixed": val_metrics_fixed,
            "val_metrics_scan": val_metrics_show if thr_scan else None,

            "args": {
                "model": "wetbird",
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "img_scale": img_scale,
                "amp": amp,
                "w_bce": w_bce,
                "w_dice": w_dice,
                "pos_weight": pos_weight,
                "val_objective": val_objective,
                "thr_scan": thr_scan,
                "thr_min": thr_min,
                "thr_max": thr_max,
                "thr_step": thr_step,
                "threshold_fallback": threshold_fallback,

                "scheduler_metric": scheduler_metric,
                "lr_patience": lr_patience,
                "lr_factor": lr_factor,
                "lr_threshold": lr_threshold,
                "lr_cooldown": lr_cooldown,
                "min_lr": min_lr,
            },
        }

        if hasattr(train_set, "mask_values"):
            payload["model_state_dict"]["mask_values"] = train_set.mask_values

        save_checkpoint(checkpoint_dir / "checkpoint_last.pth", payload)

        if is_best:
            save_checkpoint(checkpoint_dir / "checkpoint_best.pth", payload)

        if (epoch % checkpoint_interval == 0) or (epoch == epochs):
            save_checkpoint(checkpoint_dir / f"checkpoint_epoch{epoch}.pth", payload)

        t1 = time.time()
        if metrics_csv is not None:
            append_metrics_csv(
                csv_path=metrics_csv,
                epoch=epoch,
                train_loss=train_loss_avg,

                sched_objective=val_objective,
                sched_score=sched_score,
                sched_threshold=float(threshold_fallback),

                scan_enabled=bool(thr_scan),
                scan_score=float(scan_score),
                best_threshold=float(best_thr),

                val_metrics_for_show=val_metrics_show,
                lr=float(lr_now),
                best_score=float(best_score),
                epoch_time_sec=float(t1 - t0),
            )


def get_args():
    p = argparse.ArgumentParser(description="Train WetBirdNet (binary, single-channel) for best Dice/mIoU")

    p.add_argument("--train-img", type=str, default=DEFAULT_TRAIN_IMG)
    p.add_argument("--train-mask", type=str, default=DEFAULT_TRAIN_MSK)
    p.add_argument("--val-img", type=str, default=DEFAULT_VAL_IMG)
    p.add_argument("--val-mask", type=str, default=DEFAULT_VAL_MSK)

    p.add_argument("--epochs", "-e", type=int, default=100)
    p.add_argument("--batch-size", "-b", type=int, default=8)
    p.add_argument("--learning-rate", "-l", dest="lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--scale", "-s", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", default=True)

    p.add_argument("--classes", "-c", type=int, default=1)
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducible repeated experiments")
    p.add_argument("--deterministic", action="store_true", default=False,
                   help="Enable deterministic cuDNN behavior when --seed is set")

    p.add_argument("--w-bce", type=float, default=0.3)
    p.add_argument("--w-dice", type=float, default=0.7)

    p.add_argument("--pos-weight", type=float, default=1.0, help="<=0 disables pos_weight; try 2~6 if recall is low")

    p.add_argument("--val-objective", type=str, default="dice", choices=["dice", "iou_fg", "miou"],
                   help="Metric used for reporting & objective mapping inside evaluate()")

    # Threshold scan: records thr*. Not recommended as a scheduler driver
    # (the scheduler uses the fixed threshold by default).
    p.add_argument("--no-thr-scan", dest="thr_scan", action="store_false", help="Disable threshold scanning")
    p.set_defaults(thr_scan=True)
    p.add_argument("--thr-min", type=float, default=0.30)
    p.add_argument("--thr-max", type=float, default=0.70)
    p.add_argument("--thr-step", type=float, default=0.05)
    p.add_argument("--thr-fallback", type=float, default=0.5)

    p.add_argument("--scheduler-metric", type=str, default="fixed", choices=["fixed", "scan"],
                   help="fixed: use fallback threshold score for scheduler; scan: use thr-scan best score (noisier)")

    p.add_argument("--lr-patience", type=int, default=10)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--lr-threshold", type=float, default=1e-4)
    p.add_argument("--lr-cooldown", type=int, default=2)
    p.add_argument("--min-lr", type=float, default=1e-6)

    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints/wetbird_net")
    p.add_argument("--checkpoint-interval", type=int, default=20)
    p.add_argument("--resume", type=str, default=None)

    p.add_argument("--log-dir", type=str, default=None,
                   help="Directory to save log file. Default: <checkpoint-dir>/logs")
    p.add_argument("--run-name", type=str, default="train",
                   help="Prefix of log filename, e.g., train / exp01")

    p.add_argument("--metrics-csv", type=str, default=None,
                   help="Path to metrics.csv. Default: <checkpoint-dir>/metrics.csv")

    return p.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.seed is not None:
        set_random_seed(int(args.seed), deterministic=bool(args.deterministic))

    ckpt_dir = Path(args.checkpoint_dir)

    log_dir = Path(args.log_dir) if args.log_dir else (ckpt_dir / "logs")
    log_file = setup_logging(log_dir=log_dir, run_name=args.run_name)
    logging.info(f"Log file: {log_file}")

    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else (ckpt_dir / "metrics.csv")
    logging.info(f"Metrics CSV: {metrics_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if args.classes != 1:
        logging.warning("Binary training expects --classes=1. Forcing to 1.")
        args.classes = 1

    model = build_model(n_classes=args.classes).to(memory_format=torch.channels_last)
    model.to(device=device)
    logging.info("Model: WetBirdNet")

    pos_weight = None if float(args.pos_weight) <= 0 else float(args.pos_weight)

    try:
        train_model(
            model=model,
            device=device,
            train_img=Path(args.train_img),
            train_mask=Path(args.train_mask),
            val_img=Path(args.val_img),
            val_mask=Path(args.val_mask),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            img_scale=args.scale,
            amp=args.amp,
            w_bce=args.w_bce,
            w_dice=args.w_dice,
            pos_weight=pos_weight,
            val_objective=args.val_objective,
            thr_scan=args.thr_scan,
            thr_min=args.thr_min,
            thr_max=args.thr_max,
            thr_step=args.thr_step,
            threshold_fallback=args.thr_fallback,

            scheduler_metric=args.scheduler_metric,
            lr_patience=args.lr_patience,
            lr_factor=args.lr_factor,
            lr_threshold=args.lr_threshold,
            lr_cooldown=args.lr_cooldown,
            min_lr=args.min_lr,

            checkpoint_dir=ckpt_dir,
            checkpoint_interval=args.checkpoint_interval,
            resume_ckpt=args.resume,
            metrics_csv=metrics_csv,
            seed=args.seed,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error("OOM Error! Try reducing batch size or enabling --amp")
        torch.cuda.empty_cache()
        raise
