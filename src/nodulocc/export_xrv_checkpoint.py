"""Export TorchXRayVision checkpoints into NoduLoCC-compatible init weights.

Usage:
  python -m nodulocc.export_xrv_checkpoint \
    --weights resnet50-res512-all \
    --out artifacts/pretrained/xrv_resnet50_res512_all_init.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def _build_xrv_model(weights: str) -> Any:
    """Instantiate a TorchXRayVision model from a weights identifier."""
    try:
        import torchxrayvision as xrv
    except Exception as exc:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError(
            "torchxrayvision is required for this export. Install with: pip install torchxrayvision"
        ) from exc

    # Known pretrained families in TorchXRayVision.
    if weights.startswith("resnet"):
        return xrv.models.ResNet(weights=weights)
    if weights.startswith("densenet"):
        return xrv.models.DenseNet(weights=weights)

    # Fallback heuristic: try ResNet then DenseNet.
    try:
        return xrv.models.ResNet(weights=weights)
    except Exception:
        return xrv.models.DenseNet(weights=weights)


def _strip_prefixes(key: str, prefixes: list[str]) -> str:
    """Remove one leading prefix when present."""
    out = key
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if prefix and out.startswith(prefix):
                out = out[len(prefix) :]
                changed = True
                break
    return out


def _convert_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    strip_prefixes: list[str],
    convert_first_conv_to_rgb: bool,
) -> dict[str, torch.Tensor]:
    """Convert checkpoint keys/tensors to match timm-style backbones.

    TorchXRayVision models are generally single-channel. Our NoduLoCC pipeline
    uses 3-channel tensors, so we optionally adapt `conv1.weight` from 1->3
    channels by repeating and averaging the original filters.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = _strip_prefixes(str(key), strip_prefixes)
        tensor = value.detach().cpu()

        if (
            convert_first_conv_to_rgb
            and new_key == "conv1.weight"
            and tensor.ndim == 4
            and tensor.shape[1] == 1
        ):
            tensor = tensor.repeat(1, 3, 1, 1) / 3.0

        out[new_key] = tensor
    return out


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export TorchXRayVision weights for NoduLoCC init.")
    parser.add_argument(
        "--weights",
        type=str,
        default="resnet50-res512-all",
        help="TorchXRayVision weight id (e.g., resnet50-res512-all, densenet121-res224-all).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output checkpoint path (.pt).",
    )
    parser.add_argument(
        "--no-rgb-conv1",
        action="store_true",
        help="Disable 1->3 channel conversion for conv1.weight.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()
    model = _build_xrv_model(args.weights)
    state_dict = model.state_dict()

    converted = _convert_state_dict(
        state_dict,
        strip_prefixes=["module.", "model."],
        convert_first_conv_to_rgb=not bool(args.no_rgb_conv1),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": converted,
        "meta": {
            "source": "torchxrayvision",
            "weights": args.weights,
            "converted_conv1_to_rgb": not bool(args.no_rgb_conv1),
        },
    }
    torch.save(payload, out_path)

    print(f"Saved: {out_path}")
    print(f"Total tensors: {len(converted)}")
    print("First keys:", list(converted.keys())[:8])


if __name__ == "__main__":
    main()
