# Deployment image for the live app only (app/ + the demo corpus). The
# research pipeline, its full dependency set, and the private research
# corpus (data/) are deliberately NEVER copied into this image -- see
# requirements-deploy.txt and .dockerignore for why.
#
# Base: python:3.12-slim (Debian) -- matches the Python version this
# project has run on throughout (3.12.3, confirmed via `python3 --version`
# in this same environment).
FROM python:3.12-slim

# tesseract-ocr + tesseract-ocr-eng/osd: the exact package set verified
# working locally for app/upload/extraction.py's OCR fallback (PROGRESS.md
# records tesseract 5.3.4 confirmed via a real upload->OCR->answer test).
# --no-install-recommends means the English/orientation language data
# must be listed explicitly -- apt would otherwise skip them as mere
# Recommends, silently breaking OCR.
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      tesseract-ocr-eng \
      tesseract-ocr-osd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch from PyTorch's own wheel index -- a separate step from
# requirements-deploy.txt on purpose, see that file's header comment.
RUN pip install --no-cache-dir \
      torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Bake the embedding (bge-large-en-v1.5, ~1.3GB) and reranker
# (bge-reranker-large, ~2.2GB) model weights into the image at BUILD
# time, rather than letting the app lazily download them from Hugging
# Face Hub on the first production request (the previous behavior). A
# first request downloading ~3.5GB over the network before it can
# answer anything is a real reliability risk -- a slow/flaky network
# path, or a request timeout, on literally the first request a cold
# (min-instances=0) container ever serves. HF_HOME is set explicitly
# so the build-time download and the runtime load both resolve to the
# exact same path regardless of $HOME.
ENV HF_HOME=/opt/hf_cache
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
SentenceTransformer('BAAI/bge-large-en-v1.5', device='cpu'); \
AutoTokenizer.from_pretrained('BAAI/bge-reranker-large'); \
AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')"
# Forces strict offline mode at RUNTIME (not during the build step
# above, which still needs network to fetch the weights) -- this is
# how "loads without downloading at runtime" is actually verified, not
# just hoped for: if the cache were somehow incomplete, the app would
# fail immediately with a clear Hub-unreachable error instead of
# silently falling back to a slow download.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Only the app code and the specific research scripts the app imports at
# runtime (fusion/bm25/dense retrieval, the reranker, the Ollama prompt
# builder, metadata filtering) -- NOT the full scripts/ directory, which
# also holds research-only tooling (eval harnesses, corpus-building
# scripts, ir_datasets loaders) that has no place in a public image.
COPY app/ app/
COPY scripts/fusion.py scripts/bm25_retrieve.py scripts/dense_retrieve.py \
     scripts/rerank.py scripts/metadata_filter.py scripts/llm_client.py \
     scripts/
COPY demo_data/ demo_data/

# Demo config is the only one ever served publicly -- the private research
# corpus (data/) is never copied into this image in the first place, so
# there is no "accidentally serve the research config" failure mode here,
# but this is still the explicit, correct default for a public deployment.
ENV APP_CONFIG=demo \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Uploads/session data live outside the image by design (app/config.py's
# upload_dir) -- a real deployment should point this at the platform's
# ephemeral/tmp storage or a mounted volume, never at a path that would
# get baked into a new image layer or persist across deploys.
VOLUME ["/tmp/doc-rag-uploads"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://127.0.0.1:8000/health', timeout=3).raise_for_status()" || exit 1

# Single worker: the embedding/reranker models are loaded once, lazily,
# into process memory (app/rag_pipeline.py's lazy singletons) -- multiple
# uvicorn workers would each load their own copy, multiplying RAM use for
# no benefit on the small/sporadic traffic this app expects. Scale
# horizontally (more container instances) instead of with --workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
