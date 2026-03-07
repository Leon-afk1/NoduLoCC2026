"""Utility to preview preprocessing outputs for NIH and LIDC samples.

Usage:
    python -m nodulocc.preview_preprocessing --config configs/classification.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch

from .config import load_config
from .data import (
    ClassificationDataset,
    _augmentation_settings,
    _clahe_settings,
    _normalization_stats,
    _preprocessing_profile,
    load_classification_dataframe,
)


def _read_original(image_path: str, display_size: int) -> Image.Image:
    """Read an image from disk and convert it to display RGB.

    Handles both 8-bit and 16-bit grayscale sources with robust intensity
    scaling for visualization.
    """
    with Image.open(image_path) as img:
        arr = np.asarray(img)

    if arr.ndim == 3:
        # Keep display path simple: convert RGB-like inputs to grayscale first.
        arr = arr.mean(axis=2)

    arr_f = arr.astype(np.float32)
    p_low, p_high = np.percentile(arr_f, [1.0, 99.0])
    if p_high <= p_low:
        p_high = p_low + 1e-6
    arr_f = np.clip(arr_f, p_low, p_high)
    arr_u8 = ((arr_f - p_low) / (p_high - p_low + 1e-6) * 255.0).astype(np.uint8)

    gray = Image.fromarray(arr_u8, mode="L")
    resized = gray.resize((display_size, display_size), resample=Image.Resampling.BILINEAR)
    return resized.convert("RGB")


def _tensor_to_display(
    tensor: torch.Tensor,
    normalization: tuple[list[float], list[float]] | None,
    display_size: int,
) -> Image.Image:
    """Convert a model-input tensor to displayable RGB image."""
    x = tensor.detach().cpu().float().clone()
    if normalization is not None:
        mean, std = normalization
        mean_t = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
        std_t = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
        x = x * std_t + mean_t
    x = x.clamp(0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    img = Image.fromarray(arr, mode="RGB")
    return img.resize((display_size, display_size), resample=Image.Resampling.BILINEAR)


def _build_grid(rows: list[dict[str, Any]], display_size: int, title: str) -> Image.Image:
    """Build a side-by-side original/preprocessed grid."""
    margin = 10
    title_h = 24
    caption_h = 18
    width = margin * 3 + display_size * 2
    height = margin + title_h + (display_size + caption_h + margin) * max(1, len(rows))
    canvas = Image.new("RGB", (width, height), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, margin), title, fill=(235, 235, 235))
    draw.text((margin, margin + 12), "left: original | right: preprocessed", fill=(180, 180, 180))

    for i, row in enumerate(rows):
        y = margin + title_h + i * (display_size + caption_h + margin)
        canvas.paste(row["original"], (margin, y))
        canvas.paste(row["processed"], (margin * 2 + display_size, y))
        caption = f'{row["file_name"]} | y={row["label"]}'
        draw.text((margin, y + display_size + 2), caption[:90], fill=(220, 220, 220))

    return canvas


def _preview_source(
    cfg: dict[str, Any],
    source_df: pd.DataFrame,
    source_name: str,
    n_per_source: int,
    seed: int,
    display_size: int,
    train_view: bool,
) -> tuple[Image.Image | None, list[str]]:
    """Create one preview grid for a specific source."""
    if source_df.empty:
        return None, []

    sample_n = min(n_per_source, len(source_df))
    sample_df = source_df.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    normalization = _normalization_stats(cfg)
    dataset = ClassificationDataset(
        sample_df,
        img_size=int(cfg["train"].get("img_size", 512)),
        train=train_view,
        normalization=normalization,
        clahe=_clahe_settings(cfg),
        augmentation_settings=_augmentation_settings(cfg),
        preprocessing_profile=_preprocessing_profile(cfg),
        cfg=cfg,
    )

    rows: list[dict[str, Any]] = []
    selected_files: list[str] = []
    for idx in range(len(sample_df)):
        item = dataset[idx]
        file_name = str(item["file_name"])
        selected_files.append(file_name)
        rows.append(
            {
                "file_name": file_name,
                "label": int(float(item["label"])),
                "original": _read_original(str(sample_df.iloc[idx]["image_path"]), display_size=display_size),
                "processed": _tensor_to_display(item["image"], normalization=normalization, display_size=display_size),
            }
        )

    title = f"source={source_name} | profile={_preprocessing_profile(cfg)} | train_view={train_view}"
    return _build_grid(rows, display_size=display_size, title=title), selected_files


def main() -> None:
    """Generate preprocessing preview images and metadata."""
    parser = argparse.ArgumentParser(description="Preview classification preprocessing for NIH and LIDC sources.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--out-dir", type=str, default="artifacts/preprocess_preview", help="Output directory.")
    parser.add_argument("--n-per-source", type=int, default=6, help="Number of samples per source.")
    parser.add_argument("--display-size", type=int, default=256, help="Tile size for output image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for sample selection.")
    parser.add_argument(
        "--train-view",
        action="store_true",
        help="If set, apply train-time augmentations before rendering previews.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override(s) as key=value, can be repeated.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=list(args.override))
    df = load_classification_dataframe(cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "config": str(cfg.get("_config_path", args.config)),
        "profile": _preprocessing_profile(cfg),
        "train_view": bool(args.train_view),
        "n_per_source": int(args.n_per_source),
        "seed": int(args.seed),
        "outputs": {},
    }

    for source_name in ["NIH", "LIDC"]:
        source_df = df[df["source"] == source_name].reset_index(drop=True)
        grid, selected_files = _preview_source(
            cfg=cfg,
            source_df=source_df,
            source_name=source_name,
            n_per_source=int(args.n_per_source),
            seed=int(args.seed),
            display_size=int(args.display_size),
            train_view=bool(args.train_view),
        )
        if grid is None:
            continue
        out_file = out_dir / f"{source_name.lower()}_preview.png"
        grid.save(out_file)
        metadata["outputs"][source_name] = {
            "preview_file": str(out_file),
            "count": len(selected_files),
            "files": selected_files,
        }
        print(f"[preview] wrote {out_file} ({len(selected_files)} samples)")

    metadata_path = out_dir / "preview_index.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[preview] wrote {metadata_path}")


if __name__ == "__main__":
    main()
