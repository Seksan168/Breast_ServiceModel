# Breast Lesion Segmentation API — CPU image for the AWS path.
#
# There is no free GPU on AWS, so this installs the CPU-only torch wheels and
# runs on any instance (e.g. a t3.large you start/stop to conserve credits).
#
# IMPORTANT: build from the PROJECT ROOT ("Project Breast"), not from api/, so
# the build context can see both api/ and Segmentation/:
#
#     # 1) slim the checkpoints first (2.9 GB -> ~1 GB, weights only)
#     Segmentation\.venv\Scripts\python api\shrink_checkpoints.py
#     # 2) build + run
#     docker build -f api/Dockerfile -t breast-api .
#     docker run --rm -p 8000:8000 breast-api
#
# Then: http://localhost:8000/docs
FROM python:3.12-slim

# libgomp1: OpenMP runtime torch needs; libglib2.0-0: opencv (albumentations) needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (cached layer). CPU torch wheels come from PyTorch's CPU index.
COPY api/requirements-docker.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-docker.txt

# App code + the model definitions it imports (model.py pulls in swin_unet.py).
COPY api/serve_api.py .
COPY Segmentation/model.py Segmentation/swin_unet.py ./

# Trained weights — the slim, weights-only copies. Run shrink_checkpoints.py first!
COPY Segmentation/checkpoints_slim ./checkpoints_slim

# model.py + checkpoints_slim now sit under /app, so point the app here.
ENV MODELS_ROOT=/app \
    NO_ALBUMENTATIONS_UPDATE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "serve_api:app", "--host", "0.0.0.0", "--port", "8000"]
