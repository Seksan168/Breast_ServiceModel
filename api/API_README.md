# Breast Lesion Segmentation API

FastAPI service exposing the 6 best models (ranked by test Dice — see
`../Segmentation/bestmodelforapi.md`). Models load lazily on first use and are
cached; only the models you call occupy memory.

This folder (`api/`) is self-contained for deployment. The model **definitions**
(`model.py`, `swin_unet.py`) and **weights** (`checkpoints/`) live in
`../Segmentation`; the app finds them automatically when run in-repo, and the
Docker image copies them in.

## Files

| File | Purpose |
|------|---------|
| `serve_api.py` | FastAPI app — endpoints, 6-model registry, inference |
| `shrink_checkpoints.py` | Strip optimizer state → weights only (~⅓ size) |
| `requirements-api.txt` | API-only deps (add to the existing training venv) |
| `requirements-docker.txt` | Full pinned deps for the container (CPU torch) |
| `Dockerfile` | CPU image for the AWS path |
| `../.dockerignore` | Keeps the build context small (excludes venv, full ckpts, data) |

## A. Run locally (uses your GPU)

The training venv `../Segmentation/.venv` already has torch / smp / albumentations.
Add the API deps once:

```powershell
..\Segmentation\.venv\Scripts\python -m pip install -r requirements-api.txt
```

(Recommended) slim the checkpoints — 2.9 GB → ~1 GB, weights only. Originals are
left untouched; the API loads the slim copies automatically.

```powershell
..\Segmentation\.venv\Scripts\python shrink_checkpoints.py          # the 6 API models
..\Segmentation\.venv\Scripts\python shrink_checkpoints.py --all    # every best_*.pth
```

Serve:

```powershell
..\Segmentation\.venv\Scripts\python -m uvicorn serve_api:app --host 0.0.0.0 --port 8000
```

Docs UI (test uploads in the browser): <http://localhost:8000/docs>

Expose it for free with a tunnel (keeps GPU speed, $0):

```powershell
cloudflared tunnel --url http://localhost:8000   # or: ngrok http 8000
```

## B. Docker (the AWS path, CPU)

No free GPU on AWS, so the image installs CPU-only torch and runs on any instance.
**Build from the project root** so the context sees both `api/` and `Segmentation/`:

```powershell
cd "D:\Project Breast"
# 1) slim the checkpoints first — the image only copies checkpoints_slim/
Segmentation\.venv\Scripts\python api\shrink_checkpoints.py
# 2) build + run
docker build -f api/Dockerfile -t breast-api .
docker run --rm -p 8000:8000 breast-api
```

### Deploy to AWS (EC2 + ECR, on free credits)

```bash
# Push the image to ECR
aws ecr create-repository --repository-name breast-api
aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
docker tag breast-api <acct>.dkr.ecr.<region>.amazonaws.com/breast-api:latest
docker push <acct>.dkr.ecr.<region>.amazonaws.com/breast-api:latest

# On a t3.large (8 GB) EC2 with Docker + port 8000 open in the security group:
docker run -d -p 8000:8000 <acct>.dkr.ecr.<region>.amazonaws.com/breast-api:latest
```

> Cost tip: **stop the instance when idle** — you're billed per second while it
> runs, so $100 of credits lasts a long time. CPU inference is ~1–3 s/image.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info + model list |
| GET | `/models` | Registry: arch, encoder, test Dice/F1, tuned threshold |
| GET | `/health` | Device + loaded models (used by the container healthcheck) |
| POST | `/predict` | One model (default `best`) |
| POST | `/predict_all` | All 6 models on one image |
| POST | `/ensemble` | Average selected models (most accurate) |

### Query params (predict / ensemble)

| Param | Default | Values |
|-------|---------|--------|
| `model` | `best` | any key from `/models` |
| `threshold` | model's tuned value | `0.0`–`1.0` |
| `mode` | `resize` | `resize` (fast, 1 pass) · `sliding` (full-res, slower) |
| `return` | `json` | `json` · `mask` (PNG) · `overlay` (PNG) · `prob` (PNG) |

Model keys: `best`, `linknet_seresnext50`, `manet_mitb2`, `swinunet`,
`unetpp_resnet50`, `best_f1`.

## Examples (curl)

```bash
# Best model, JSON stats (area %, bbox, centroid)
curl -F "file=@mammo.png" "http://localhost:8000/predict?model=best"

# Overlay PNG back
curl -F "file=@mammo.png" "http://localhost:8000/predict?return=overlay" -o overlay.png

# Compare all 6 models
curl -F "file=@mammo.png" "http://localhost:8000/predict_all"

# Ensemble (most accurate), custom threshold + binary mask PNG
curl -F "file=@mammo.png" "http://localhost:8000/ensemble?threshold=0.35&return=mask" -o mask.png
```

The app auto-detects CUDA; on a CPU host (Docker/AWS) it runs unchanged, just slower.
