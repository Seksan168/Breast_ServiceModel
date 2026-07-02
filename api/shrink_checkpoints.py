"""
shrink_checkpoints.py — strip optimizer/scheduler/scaler state from checkpoints.

Training checkpoints (see model.save_checkpoint) store the optimizer, scheduler
and AMP-scaler state alongside the weights. That roughly DOUBLES the file size
(e.g. 585 MB instead of ~290 MB for a 51 M-param model) and is useless for
inference — the API only ever calls ``load_state_dict`` on the model weights.

This script re-saves each checkpoint keeping ONLY ``model_state_dict`` (plus the
epoch / dice for provenance) into a mirror folder ``checkpoints_slim/`` with the
same sub-paths, so the originals are never touched. ``serve_api.py`` prefers the
slim copy automatically when it exists.

Usage
-----
    # Slim just the 6 models the API serves (default):
    python shrink_checkpoints.py

    # Slim every best_*.pth under checkpoints/:
    python shrink_checkpoints.py --all

    # Custom output dir:
    python shrink_checkpoints.py --out checkpoints_slim
"""

import argparse
import glob
import os
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
# Model code + checkpoints live in ../Segmentation (this script sits in api/).
SEG_DIR = SCRIPT_DIR.parent / "Segmentation"
CKPT_DIR = SEG_DIR / "checkpoints"

# The 6 checkpoints served by the API (relative to checkpoints/).
API_MODELS = [
    "unetpp_se_resnext50_32x4d_aug/best_dice0.5395_bs8.pth",
    "linknet_se_resnext50_32x4d_aug/best_dice0.5454_bs8.pth",
    "manet_mit_b2_aug/best_dice0.5771_bs8.pth",
    "swinunet_swin_tiny_aug/best_dice0.5743_bs4.pth",
    "unetpp_resnet50_aug/best_dice0.5360_bs8.pth",
    "unetpp_se_resnext50_32x4d_aug_ftn/best_dice0.5596_bs8.pth",
]


def _mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def slim_one(src: Path, dst: Path) -> tuple[float, float]:
    """Write a weights-only copy of ``src`` to ``dst``. Returns (src_mb, dst_mb)."""
    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    slim = {
        "model_state_dict": ckpt["model_state_dict"],
        "epoch": ckpt.get("epoch"),
        "metric_value": ckpt.get("metric_value"),
        "slim": True,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dst)
    return _mb(src), _mb(dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="Slim every best_*.pth under checkpoints/ (not just the 6 API models)")
    ap.add_argument("--out", default="checkpoints_slim",
                    help="Output mirror directory (default: checkpoints_slim)")
    args = ap.parse_args()

    out_root = SEG_DIR / args.out

    if args.all:
        srcs = [Path(p) for p in glob.glob(str(CKPT_DIR / "**" / "best_*.pth"), recursive=True)]
    else:
        srcs = [CKPT_DIR / rel for rel in API_MODELS]

    missing = [s for s in srcs if not s.exists()]
    for m in missing:
        print(f"[skip] not found: {m}")
    srcs = [s for s in srcs if s.exists()]
    if not srcs:
        print("No checkpoints to process."); return

    total_src = total_dst = 0.0
    print(f"Slimming {len(srcs)} checkpoint(s) -> {out_root}\n")
    for src in srcs:
        rel = src.relative_to(CKPT_DIR)
        dst = out_root / rel
        s_mb, d_mb = slim_one(src, dst)
        total_src += s_mb; total_dst += d_mb
        print(f"  {rel}\n    {s_mb:7.1f} MB -> {d_mb:7.1f} MB  "
              f"(saved {s_mb - d_mb:6.1f} MB, {100 * (1 - d_mb / s_mb):4.1f}%)")

    print(f"\nTotal: {total_src:.1f} MB -> {total_dst:.1f} MB  "
          f"(saved {total_src - total_dst:.1f} MB, {100 * (1 - total_dst / total_src):.1f}%)")
    print(f"Slim checkpoints written under: {out_root}")
    print("serve_api.py will now load these automatically.")


if __name__ == "__main__":
    main()
