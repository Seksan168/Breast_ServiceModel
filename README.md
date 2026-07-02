# Breast Lesion Segmentation — Service Model API

FastAPI service that serves the 6 best breast-lesion segmentation models
(ranked by test Dice). Upload a mammogram, get back the predicted lesion
location (JSON stats or an overlay/mask PNG).

## Repo layout

```
api/                     # the service (deployment artifact)
  serve_api.py           #   FastAPI app — endpoints + 6-model registry
  shrink_checkpoints.py  #   strip optimizer state from checkpoints (~⅔ smaller)
  Dockerfile             #   CPU image (the AWS path)
  requirements-*.txt     #   deps (local / container)
  API_README.md          #   full endpoint + deploy docs
Segmentation/
  model.py, swin_unet.py # model definitions the API imports
  bestmodelforapi.md     # how the 6 models were chosen (test Dice / F1 ranking)
.dockerignore            # keeps the build context to the 6 served models
```

> **Model weights are not stored in git** (too large). Produce them with the
> training pipeline, then run `api/shrink_checkpoints.py` to create the slim,
> weights-only copies the service and Docker image use. See `api/API_README.md`.

## Quickstart

```bash
# Local (uses your GPU if available)
pip install -r api/requirements-api.txt
python -m uvicorn serve_api:app --app-dir api --host 0.0.0.0 --port 8000
# → http://localhost:8000/docs

# Docker (CPU — the AWS path); build from the repo root
docker build -f api/Dockerfile -t breast-api .
docker run --rm -p 8000:8000 breast-api
```

Full endpoint reference, curl examples, and AWS (ECR + EC2) deploy steps live in
[`api/API_README.md`](api/API_README.md).

## Models served

| Key | Architecture | Test Dice | F1 |
|-----|--------------|:--:|:--:|
| `best` | U-Net++ + se_resnext50_32x4d (aug) | 0.5853 | 0.6636 |
| `linknet_seresnext50` | LinkNet + se_resnext50_32x4d (aug) | 0.5818 | 0.6600 |
| `manet_mitb2` | MAnet + mit_b2 (aug) | 0.5748 | 0.4474 |
| `swinunet` | Swin-Unet + swin_tiny (aug) | 0.5747 | 0.6282 |
| `unetpp_resnet50` | U-Net++ + resnet50 (aug) | 0.5718 | 0.6560 |
| `best_f1` | U-Net++ + se_resnext50_32x4d (aug, ftn) | 0.5707 | 0.6765 |
