"""
serve_api.py — FastAPI service for the breast-lesion segmentation models.

Serves the 6 best models (ranked by test Dice — see bestmodelforapi.md) behind a
REST API. Models are built from `model.build_model` and load their trained
weights on first use (lazy) so startup is instant and only the models you call
occupy memory. If `checkpoints_slim/` exists (see shrink_checkpoints.py) the slim
weights are loaded automatically.

Endpoints
---------
    GET  /                      — service info + model list
    GET  /models                — registry metadata (arch, encoder, dice, f1, thr)
    GET  /health                — liveness + loaded models + device
    POST /predict               — one model (default: best). Upload an image.
    POST /predict_all           — run all 6 models on one image, compare.
    POST /ensemble              — average selected models (most accurate).

Upload field is `file` (multipart). Common query params:
    model       model key (see /models); default "best"
    threshold   override the model's tuned threshold (0..1)
    mode        "resize" (fast, 1 pass) | "sliding" (full-res, slower)
    return      "json" (stats) | "mask" (PNG) | "overlay" (PNG) | "prob" (PNG)

Run
---
    pip install -r requirements-api.txt
    uvicorn serve_api:app --host 0.0.0.0 --port 8000
    # docs UI at http://localhost:8000/docs
"""

import io
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from fastapi import FastAPI, File, Query, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ── Locate the model code + checkpoints ───────────────────────────────────────
# Works both in-repo (this file in api/, model code in ../Segmentation) and in
# the Docker image (model.py copied next to this file, MODELS_ROOT=/app).
SCRIPT_DIR = Path(__file__).resolve().parent
SEG_DIR = SCRIPT_DIR if (SCRIPT_DIR / "model.py").exists() else SCRIPT_DIR.parent / "Segmentation"
sys.path.insert(0, str(SEG_DIR))

from model import build_model  # noqa: E402 — needs SEG_DIR on sys.path first

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", str(SEG_DIR)))
CKPT_DIR = MODELS_ROOT / "checkpoints"
SLIM_DIR = MODELS_ROOT / "checkpoints_slim"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512  # all models were trained at 512

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_NORM = A.Compose([A.Normalize(mean=_MEAN, std=_STD), ToTensorV2()])

# ──────────────────────────────────────────────────────────────────────────────
# Model registry — the 6 models from bestmodelforapi.md, ranked by test Dice.
# `arch` is the smp architecture name build_model expects; `encoder` is the
# backbone; `threshold` is the val-tuned threshold from each model's report.
# ──────────────────────────────────────────────────────────────────────────────
MODELS: dict[str, dict] = {
    "best": {  # alias → rank 1
        "rank": 1, "arch": "UnetPlusPlus", "encoder": "se_resnext50_32x4d",
        "ckpt": "unetpp_se_resnext50_32x4d_aug/best_dice0.5395_bs8.pth",
        "threshold": 0.45, "test_dice": 0.5853, "test_f1": 0.6636,
        "label": "U-Net++ + se_resnext50_32x4d (aug)",
    },
    "linknet_seresnext50": {
        "rank": 2, "arch": "Linknet", "encoder": "se_resnext50_32x4d",
        "ckpt": "linknet_se_resnext50_32x4d_aug/best_dice0.5454_bs8.pth",
        "threshold": 0.50, "test_dice": 0.5818, "test_f1": 0.6600,
        "label": "LinkNet + se_resnext50_32x4d (aug)",
    },
    "manet_mitb2": {
        "rank": 3, "arch": "MAnet", "encoder": "mit_b2",
        "ckpt": "manet_mit_b2_aug/best_dice0.5771_bs8.pth",
        "threshold": 0.30, "test_dice": 0.5748, "test_f1": 0.4474,
        "label": "MAnet + mit_b2 (aug)  [high Dice, low F1 — see report]",
    },
    "swinunet": {
        "rank": 4, "arch": "SwinUnet", "encoder": "swin_tiny",
        "ckpt": "swinunet_swin_tiny_aug/best_dice0.5743_bs4.pth",
        "threshold": 0.05, "test_dice": 0.5747, "test_f1": 0.6282,
        "label": "Swin-Unet + swin_tiny (aug)",
    },
    "unetpp_resnet50": {
        "rank": 5, "arch": "UnetPlusPlus", "encoder": "resnet50",
        "ckpt": "unetpp_resnet50_aug/best_dice0.5360_bs8.pth",
        "threshold": 0.05, "test_dice": 0.5718, "test_f1": 0.6560,
        "label": "U-Net++ + resnet50 (aug)",
    },
    "best_f1": {  # rank 6 — highest F1
        "rank": 6, "arch": "UnetPlusPlus", "encoder": "se_resnext50_32x4d",
        "ckpt": "unetpp_se_resnext50_32x4d_aug_ftn/best_dice0.5596_bs8.pth",
        "threshold": 0.25, "test_dice": 0.5707, "test_f1": 0.6765,
        "label": "U-Net++ + se_resnext50_32x4d (aug, ftn) — best F1",
    },
}

# lazy weight cache: key -> nn.Module
_LOADED: dict[str, torch.nn.Module] = {}


def _resolve_ckpt(rel: str) -> Path:
    """Prefer the slim weights-only copy if present, else the full checkpoint."""
    slim = SLIM_DIR / rel
    return slim if slim.exists() else CKPT_DIR / rel


def _make_cfg(spec: dict) -> dict:
    """Minimal cfg for build_model — encoder_weights=None: no ImageNet download,
    the checkpoint provides all weights."""
    return {
        "model": {
            "architecture": spec["arch"],
            "encoder_name": spec["encoder"],
            "encoder_weights": None,
            "in_channels": 3,
            "num_classes": 1,
            "activation": None,
        },
        "data": {"image_size": IMAGE_SIZE, "patch_size": IMAGE_SIZE},
    }


def get_model(key: str) -> torch.nn.Module:
    """Build + load (once) and cache the model for `key`."""
    if key not in MODELS:
        raise HTTPException(404, f"unknown model '{key}'. See /models.")
    if key in _LOADED:
        return _LOADED[key]
    spec = MODELS[key]
    ckpt_path = _resolve_ckpt(spec["ckpt"])
    if not ckpt_path.exists():
        raise HTTPException(500, f"checkpoint missing: {ckpt_path}")
    model = build_model(_make_cfg(spec)).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    _LOADED[key] = model
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def _read_image(raw: bytes) -> np.ndarray:
    try:
        return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception as e:
        raise HTTPException(400, f"could not read image: {e}")


def predict_resize(model: torch.nn.Module, image: np.ndarray) -> np.ndarray:
    """Resize→forward→resize back. One GPU call; returns prob map at orig size."""
    H, W = image.shape[:2]
    resized = np.array(Image.fromarray(image).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
    tensor = _NORM(image=resized)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad(), torch.amp.autocast(DEVICE, enabled=(DEVICE == "cuda")):
        prob = torch.sigmoid(model(tensor))[0, 0].float().cpu().numpy()
    prob_img = Image.fromarray(prob.astype(np.float32), mode="F").resize((W, H), Image.BILINEAR)
    return np.asarray(prob_img, dtype=np.float32)


def predict_sliding(model: torch.nn.Module, image: np.ndarray,
                    window: int = IMAGE_SIZE, stride: int = IMAGE_SIZE // 2) -> np.ndarray:
    """Overlapping sliding window at full resolution (slower, more detail)."""
    H, W = image.shape[:2]
    pad_h, pad_w = max(0, window - H), max(0, window - W)
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    pH, pW = image.shape[:2]
    prob_sum = np.zeros((pH, pW), np.float32)
    count = np.zeros((pH, pW), np.float32)
    ys = sorted(set(list(range(0, pH - window, stride)) + [pH - window]))
    xs = sorted(set(list(range(0, pW - window, stride)) + [pW - window]))
    with torch.no_grad(), torch.amp.autocast(DEVICE, enabled=(DEVICE == "cuda")):
        for y in ys:
            for x in xs:
                patch = image[y:y + window, x:x + window]
                t = _NORM(image=patch)["image"].unsqueeze(0).to(DEVICE)
                p = torch.sigmoid(model(t))[0, 0].float().cpu().numpy()
                prob_sum[y:y + window, x:x + window] += p
                count[y:y + window, x:x + window] += 1.0
    return (prob_sum / np.maximum(count, 1e-6))[:H, :W]


def _prob_map(key: str, image: np.ndarray, mode: str) -> np.ndarray:
    model = get_model(key)
    return predict_sliding(model, image) if mode == "sliding" else predict_resize(model, image)


# ──────────────────────────────────────────────────────────────────────────────
# Response builders
# ──────────────────────────────────────────────────────────────────────────────

def _stats(prob: np.ndarray, threshold: float) -> dict:
    mask = prob >= threshold
    pos = int(mask.sum())
    H, W = mask.shape
    out = {
        "detected": bool(pos > 0),
        "threshold": round(float(threshold), 4),
        "lesion_area_pct": round(float(mask.mean() * 100), 4),
        "positive_pixels": pos,
        "image_size": {"height": H, "width": W},
        "prob_max": round(float(prob.max()), 4),
    }
    if pos > 0:
        ys, xs = np.where(mask)
        out["bbox"] = {"x_min": int(xs.min()), "y_min": int(ys.min()),
                       "x_max": int(xs.max()), "y_max": int(ys.max())}
        out["centroid"] = {"x": int(round(xs.mean())), "y": int(round(ys.mean()))}
    return out


def _png(arr: np.ndarray) -> StreamingResponse:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


def _overlay_png(image: np.ndarray, mask: np.ndarray) -> StreamingResponse:
    ov = image.astype(np.float32)
    ov[mask] = ov[mask] * 0.45 + np.array([220, 30, 30], np.float32) * 0.55
    return _png(ov.clip(0, 255).astype(np.uint8))


def _render(image: np.ndarray, prob: np.ndarray, threshold: float, return_: str):
    if return_ == "mask":
        return _png(((prob >= threshold).astype(np.uint8)) * 255)
    if return_ == "overlay":
        return _overlay_png(image, prob >= threshold)
    if return_ == "prob":
        return _png((prob * 255).clip(0, 255).astype(np.uint8))
    return None  # json handled by caller


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Breast Lesion Segmentation API",
    description="6 segmentation models (ranked by test Dice) — see bestmodelforapi.md",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def root():
    return {
        "service": "Breast Lesion Segmentation API",
        "device": DEVICE,
        "slim_weights": SLIM_DIR.exists(),
        "default_model": "best",
        "models": {k: v["label"] for k, v in MODELS.items()},
        "docs": "/docs",
    }


@app.get("/models")
def list_models():
    return {k: {**{kk: vv for kk, vv in v.items() if kk != "ckpt"},
                "loaded": k in _LOADED} for k, v in MODELS.items()}


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "loaded": list(_LOADED)}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query("best", description="model key — see /models"),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    mode: str = Query("resize", pattern="^(resize|sliding)$"),
    return_: str = Query("json", alias="return", pattern="^(json|mask|overlay|prob)$"),
):
    """Run one model on the uploaded image."""
    if model not in MODELS:
        raise HTTPException(404, f"unknown model '{model}'. See /models.")
    thr = MODELS[model]["threshold"] if threshold is None else threshold
    image = _read_image(await file.read())
    prob = _prob_map(model, image, mode)
    rendered = _render(image, prob, thr, return_)
    if rendered is not None:
        return rendered
    return JSONResponse({
        "model": model, "label": MODELS[model]["label"], "mode": mode,
        **_stats(prob, thr),
    })


@app.post("/predict_all")
async def predict_all(
    file: UploadFile = File(...),
    mode: str = Query("resize", pattern="^(resize|sliding)$"),
):
    """Run all 6 models on one image and return each model's stats."""
    image = _read_image(await file.read())
    results = {}
    for key, spec in sorted(MODELS.items(), key=lambda kv: kv[1]["rank"]):
        prob = _prob_map(key, image, mode)
        results[key] = {"rank": spec["rank"], "label": spec["label"],
                        **_stats(prob, spec["threshold"])}
    return {"mode": mode, "results": results}


@app.post("/ensemble")
async def ensemble(
    file: UploadFile = File(...),
    models: Optional[str] = Query(None, description="comma-separated keys; default all 6"),
    threshold: float = Query(0.35, ge=0.0, le=1.0, description="ensemble threshold (report: 0.35)"),
    mode: str = Query("resize", pattern="^(resize|sliding)$"),
    return_: str = Query("json", alias="return", pattern="^(json|mask|overlay|prob)$"),
):
    """Average the probability maps of several models (most accurate)."""
    keys = [k.strip() for k in models.split(",")] if models else list(MODELS)
    bad = [k for k in keys if k not in MODELS]
    if bad:
        raise HTTPException(404, f"unknown model(s): {bad}. See /models.")
    image = _read_image(await file.read())
    prob = np.mean([_prob_map(k, image, mode) for k in keys], axis=0)
    rendered = _render(image, prob, threshold, return_)
    if rendered is not None:
        return rendered
    return JSONResponse({"models": keys, "mode": mode, **_stats(prob, threshold)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
