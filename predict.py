"""Inference script for WetBirdNet.

Predicts binary waterbird masks for a single image or a directory of images
and writes them as 8-bit PNG masks (0 = background, 255 = bird).

Example
-------
python predict.py --model checkpoints/wetbird_net/checkpoint_best.pth \
                  --input data/imgs/test --output predictions
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wetland_unet import WetBirdNet


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def preprocess(pil_img: Image.Image, scale: float) -> torch.Tensor:
    """Match the preprocessing used by utils.data_loading at training time."""
    w, h = pil_img.size
    new_w, new_h = max(int(scale * w), 1), max(int(scale * h), 1)
    pil_img = pil_img.resize((new_w, new_h), resample=Image.BICUBIC)

    img = np.asarray(pil_img)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    else:
        img = img.transpose((2, 0, 1))

    if (img > 1).any():
        img = img / 255.0

    return torch.as_tensor(img.copy()).float().contiguous()


def load_model(checkpoint: str, device: torch.device, pretrained: bool = False) -> WetBirdNet:
    state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    state_dict.pop("mask_values", None)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    if not any(k.startswith("backbone.features.") for k in state_dict.keys()):
        raise RuntimeError(
            f"{checkpoint} does not look like a WetBirdNet checkpoint "
            "(no 'backbone.features.*' keys)."
        )

    # Infer the number of output classes from the segmentation head
    n_classes = 1
    for key in ("outc.weight", "outc.conv.weight"):
        if key in state_dict and state_dict[key].dim() >= 1:
            n_classes = int(state_dict[key].shape[0])
            break

    net = WetBirdNet(n_classes=n_classes, pretrained=pretrained)
    net.load_state_dict(state_dict, strict=True)
    net.to(device=device)
    net.eval()
    return net


@torch.no_grad()
def predict_img(net: WetBirdNet, pil_img: Image.Image, device: torch.device,
                scale: float, threshold: float) -> np.ndarray:
    """Return a uint8 mask (0/255) at the original image resolution."""
    original_w, original_h = pil_img.size

    img = preprocess(pil_img, scale).unsqueeze(0).to(device=device, dtype=torch.float32)

    logits = net(img)
    probs = torch.sigmoid(logits.float()).squeeze(0).squeeze(0)  # [H, W]

    probs = torch.nn.functional.interpolate(
        probs.unsqueeze(0).unsqueeze(0),
        size=(original_h, original_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    mask = (probs > threshold).to(torch.uint8).cpu().numpy()
    return (mask * 255).astype(np.uint8)


def collect_inputs(input_path: Path) -> list:
    if input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        )
    elif input_path.is_file():
        files = [input_path]
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not files:
        raise RuntimeError(f"No images found in {input_path}")
    return files


def get_args():
    p = argparse.ArgumentParser(description="Predict waterbird masks with WetBirdNet")
    p.add_argument("--model", "-m", type=str, required=True, help="Path to the .pth checkpoint")
    p.add_argument("--input", "-i", type=str, required=True, help="Image file or directory of images")
    p.add_argument("--output", "-o", type=str, default="predictions", help="Output directory for masks")
    p.add_argument("--threshold", "-t", type=float, default=0.5, help="Probability threshold for foreground")
    p.add_argument("--scale", "-s", type=float, default=1.0, help="Input downscaling factor (0, 1]")
    p.add_argument("--pretrained", action="store_true", default=False,
                   help="Load ImageNet weights for the backbone before the checkpoint")
    p.add_argument("--no-cuda", action="store_true", default=False, help="Force CPU inference")
    p.add_argument("--save-overlay", action="store_true", default=False,
                   help="Also save an RGB overlay of the mask on the input image")
    return p.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f"Using device: {device}")

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_inputs(input_path)
    logging.info(f"Found {len(files)} image(s) in {input_path}")

    net = load_model(args.model, device, pretrained=args.pretrained)
    logging.info(f"Loaded checkpoint: {args.model}")

    for idx, img_path in enumerate(files, start=1):
        logging.info(f"[{idx}/{len(files)}] {img_path.name}")
        try:
            with Image.open(img_path) as im:
                pil_img = im.convert("RGB")

            mask = predict_img(
                net=net,
                pil_img=pil_img,
                device=device,
                scale=args.scale,
                threshold=args.threshold,
            )

            out_path = output_dir / f"{img_path.stem}.png"
            Image.fromarray(mask).save(out_path)

            if args.save_overlay:
                base = pil_img.convert("RGB")
                overlay = np.array(base).copy()
                fg = mask > 0
                overlay[fg] = (0.4 * overlay[fg] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
                Image.fromarray(overlay).save(output_dir / f"{img_path.stem}_overlay.jpg")

        except Exception as e:
            logging.error(f"Failed on {img_path.name}: {e}")

    logging.info(f"Done. Masks saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
