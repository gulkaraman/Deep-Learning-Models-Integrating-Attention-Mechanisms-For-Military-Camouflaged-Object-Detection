"""Run inference on a folder of RGB/BGR images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.data.preprocessing import preprocess_image
from src.evaluation.model_loading import load_model_for_evaluation
from src.evaluation.visualization import save_mask_preview
from src.utils.config import ExperimentConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict masks for images in a directory.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, required=True, help="Directory containing .jpg/.jpeg/.png images.")
    p.add_argument("--output", type=str, default="outputs/predictions", help="Directory to write mask PNGs.")
    p.add_argument("--save_previews", action="store_true", help="Also save side-by-side PNG previews.")
    return p.parse_args()


def iter_images(folder: Path):
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
        yield from sorted(folder.glob(ext))


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    in_dir = Path(args.input)
    if not in_dir.is_dir():
        raise FileNotFoundError(in_dir)

    model = load_model_for_evaluation(cfg, args.checkpoint)
    # ``predict`` does not require full compile, but compiling avoids TF warnings on some versions.
    model.compile(optimizer="adam", loss="binary_crossentropy")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in iter_images(in_dir):
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(img_path)
        x = preprocess_image(bgr, cfg.image_size, cfg.use_clahe)
        batch = np.expand_dims(x, axis=0).astype(np.float32)
        prob = model.predict(batch, verbose=0)[0, ..., 0]
        mask_uint8 = (prob * 255.0).clip(0, 255).astype(np.uint8)
        stem = img_path.stem
        mask_path = out_dir / f"{stem}_mask.png"
        cv2.imwrite(str(mask_path), mask_uint8)

        if args.save_previews:
            preview_path = out_dir / f"{stem}_preview.png"
            save_mask_preview(x, prob, preview_path, title=stem)


if __name__ == "__main__":
    main()
