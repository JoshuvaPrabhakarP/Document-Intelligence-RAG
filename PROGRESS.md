# Progress — Deployment & Agent Workstream

> Tracks the mentor-guided work layered on top of the research project
> (see `PROJECT_GRAPH_REPORT.md` and `README.md` for that work). This file
> covers the new `app/` layer: FastAPI, the web interface, and (in a later
> milestone) the LangGraph workflow.

## Milestone 1 — Repository audit (complete)

Findings (verified live, not assumed):
- Resume claims (Hit@10 0.694, MRR 0.618, Transformer F1 87.6–90.8%) checked against the actual `data/eval/*.json` reports — accurate, not rounded up.
- No FastAPI/UI/Docker/CI/LangGraph existed before this milestone.
- Cold-start model loading: ~13–30s; warm: retrieval 36ms, rerank 2.6s, generation 2.3s.
- Model footprint (measured, not download size): BGE-large + reranker together ≈ 1.3GB RSS CPU-only, sub-second CPU embedding latency (468ms for a short query). Qwen3 8B (Ollama) is the real resource bottleneck at 5.2GB.

## Milestone 2 — Usable application (complete)

### What was built

```
app/
├── main.py              # FastAPI app: /health, /ready, /query, /sample-questions, static UI
├── config.py             # env-based settings (pydantic-settings), "demo" vs "research" config
├── schemas.py             # request/response models
├── rag_pipeline.py        # composes the EXISTING fusion.py/rerank.py — no retrieval logic reimplemented
├── generation/
│   ├── base.py            # GenerationProvider abstract interface + GenerationError
│   └── ollama_provider.py # the only concrete provider — reuses llm_client.py's prompt/think-strip logic
└── static/index.html      # vanilla HTML/JS interface, sample questions, loading state, error display
demo_data/
├── build_synthetic_demo_corpus.py   # generates the public demo corpus
└── demo_chunks.jsonl                 # 10 synthetic receipts (committed — small, public, safe)
tests/
├── test_api.py            # 14 fast, mocked tests — what CI gates on
└── test_integration.py    # 2 real-model smoke tests — skipped if Ollama unreachable
.env.example                # every config variable, no secrets
pytest.ini
```

**No existing research code was modified.** `app/rag_pipeline.py` and `app/generation/ollama_provider.py` import `scripts/fusion.py`, `scripts/rerank.py`, and `scripts/llm_client.py` exactly as they are — this is composition, not reimplementation.

### The demo-vs-research decision (and why)

Checked licenses directly from authoritative sources, not assumed:

| Dataset | License | Source checked |
|---|---|---|
| CORD | CC-BY-4.0, redistribution OK with attribution | `clovaai/cord` GitHub repo + HF dataset card (both confirm) |
| SROIE | **Unclear** — official source (`rrc.cvc.uab.es`) gates data behind competition registration; no public-redistribution grant found. A third party's MIT-licensed GitHub repo covers only their own code, not the receipt images they don't own. | `rrc.cvc.uab.es/?ch=13&com=downloads` |
| WildReceipt | **Unclear** — license explicitly listed as "N/A" | Official MMOCR docs (`mmocr.readthedocs.io`) |

Since 2 of 3 real sources are unclear, **the public demo uses 10 hand-authored synthetic receipts** (`demo_data/demo_chunks.jsonl`), each tagged `"source": "synthetic_demo"`. The real 3,712-document research corpus (`data/`) is untouched, stays local, and `APP_CONFIG=research` exists in config only for local/offline use — it is never the default and is not deployed.

**No research metrics are attached to the demo config.** The demo has not been evaluated the way the research system was (Steps 4–6) — it's a different, smaller, unevaluated corpus, and the UI/API say so.

### Generation: did the system already answer questions, or only retrieve?

Checked: `scripts/llm_client.py::generate_answer()` already existed from Step 4 (used inside the eval harness, not exposed as a service). **Reused, not reimplemented** — `app/generation/ollama_provider.py` imports its `build_prompt`/`strip_thinking` functions directly and adds what the research script didn't need: configurable base_url/model/timeout and `GenerationError` wrapping so API callers get a clean error instead of a raw exception.

### "Insufficient evidence" handling

No confidence threshold was ever established in the Step 4–6 research — inventing one now would be an unvalidated guess presented as a metric. Instead, this reuses the **already-existing prompt instruction** (`llm_client.py`'s prompt already tells the model to say "I don't know" when the context doesn't contain the answer) and detects that phrase. Verified for real: an off-topic question ("What is the capital of France?") correctly returned `has_sufficient_evidence: false`.

### Verification — commands actually run, with real output

```bash
# server
cp .env.example .env
venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000

# health/readiness
curl http://127.0.0.1:8000/health
# -> {"status":"ok"}
curl http://127.0.0.1:8000/ready
# -> {"ready":true,"config":"demo","checks":{"index_files_present":true,"generation_provider_reachable":true}}

# real end-to-end query
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
  -d '{"question": "What is the total amount on the Example Grocers Inc. receipt?", "top_k": 3}'
# -> answer: "The total amount on the Example Grocers Inc. receipt is $42.17." (correct)
# -> sources[0].chunk_id: "demo_000" (correct document)
# -> has_sufficient_evidence: true

# insufficient-evidence path
curl -X POST .../query -d '{"question": "What is the capital of France?"}'
# -> answer: "I don't know.", has_sufficient_evidence: false

# validation
curl -X POST .../query -d '{"question": ""}'        # -> 422
curl -X POST .../query -d '{"question": "aaa...600 chars"}'  # -> 422

# web UI
curl http://127.0.0.1:8000/   # -> 200, text/html

# full test suite
venv/bin/pytest tests/ -v
# -> 16 passed, 3 warnings in 33.40s (includes 2 REAL, non-mocked integration tests)
```

### Known limitations (honest, not glossed over)

- **Cold start is real and unmitigated.** First query after Ollama's idle timeout costs ~15–30s. Not solved in this milestone — a deployment decision for Milestone 5 (keep-alive config, a warm-up endpoint, or accepting the latency with a UI loading message, which the current interface already does).
- **No rate limiting / concurrency limits yet** — deliberately deferred to Milestone 5 ("public demo controls"), since this isn't deployed yet.
- **Hosting platform not yet chosen.** Preliminary research (Hugging Face Spaces: 16GB RAM/2 CPU on free "CPU Basic" tier, but compute-backed Spaces may require a paid plan per community sources, not yet confirmed against primary docs) — to be finalized in Milestone 5, not assumed here.
- **`requirements.txt` as committed is research-only** (CUDA-pinned torch, includes training/eval-only packages like `ragas`/`FlagEmbedding`). A trimmed, deployment-appropriate requirements file is Milestone 5 work, not done yet.
- **The demo corpus is intentionally tiny (10 receipts)** — enough to prove the system works end-to-end, not a meaningful evaluation sample. No retrieval-quality claims are made about it.

### Files changed/added this milestone

New: `app/` (8 files), `demo_data/build_synthetic_demo_corpus.py`, `demo_data/demo_chunks.jsonl`, `tests/test_api.py`, `tests/test_integration.py`, `.env.example`, `pytest.ini`, `PROGRESS.md`.
Modified: `.gitignore` (added `demo_data/index/`).
Untouched: every file under `scripts/`, all research data, `PROJECT_GRAPH_REPORT.md`, `README.md`.

**Not started: LangGraph (Milestone 3), Docker/CI/deployment (Milestone 5), documentation of a public URL (Milestone 6) — per scope, paused here.**

## Milestone 2 extension — Upload your own receipt (complete)

Adds a second, fully isolated mode to the same application: upload a real JPG/PNG/PDF receipt and ask questions about it, without touching the research pipeline or its evaluation claims.

### What was built

```
app/upload/
├── extraction.py       # PDF text-layer + OCR fallback (PyMuPDF + Tesseract), timeout-wrapped
├── session.py           # SessionManager — per-browser-session storage, lazy TTL expiry
├── indexing.py          # builds a session-scoped FAISS+BM25 index (new code — see why below)
└── retriever_cache.py   # caches one HybridRetriever (UNMODIFIED scripts/fusion.py) per session
```
Plus: `app/main.py` gained `/upload`, `/session` (GET/DELETE), session-cookie handling, and a `mode` field on `/query`; `app/rag_pipeline.py` gained `answer_from_retriever()` (same rerank/generate/format logic as `answer()`, parameterized by retriever — this is how upload mode reuses generation without reimplementing it); `app/static/index.html` gained a mode tab, upload control, extracted-text panel, and "Clear uploaded receipts."

**Retrieval (`fusion.py`) and generation (`llm_client.py`/`ollama_provider.py`) are reused exactly as before — zero changes.** The one genuinely new piece is `indexing.py`: `embed_index.py` is a CLI-only script with no reusable function, and shelling out to it per upload would force a ~13s cold model-load on every single file. `indexing.py` follows its exact pattern (same model, same `IndexFlatIP`, same `metadata.jsonl`/`index_config.json` shape) but keeps one embedding model warm across uploads instead.

### A real blocker, handled transparently

This sandbox has no passwordless `sudo`, so I could not install the system `tesseract-ocr` package myself. **You ran the install yourself** (per your choice when asked) while I built everything else in parallel. Until it's confirmed installed:
- PDF embedded-text extraction works fully and was verified end-to-end (see below) — most PDFs have a text layer and don't need OCR at all.
- Image uploads and scanned PDF pages correctly return a clean `503` with the message *"OCR engine (tesseract) is not installed on the server..."* rather than crashing — this error path was verified for real, not assumed.
- **OCR extraction accuracy itself has not been verified end-to-end** — that's the one remaining unverified claim in this milestone. Once tesseract is confirmed installed (`which tesseract`), re-upload a JPG/PNG to confirm.

### Resource decision: upload's embedding model runs on CPU, not GPU

Demo mode's retriever (GPU, ~1.3GB VRAM) + reranker (GPU, ~2.2GB) + Qwen3 8B via Ollama (~5.2GB) already approach this machine's 8GB VRAM. Loading a *second* GPU copy of the embedding model for uploads risked exceeding it. Measured earlier (Milestone 1/2): CPU embedding costs ~468ms for a short query — negligible for a one-time per-upload indexing step, so `UPLOAD_EMBEDDING_DEVICE=cpu` by default, configurable.

### Session isolation — how it actually holds, not just claimed

- Session ID is an opaque, server-generated UUID in an HTTP-only cookie — the client never supplies or influences it.
- Each session's chunks/index live in their own directory under `/tmp/doc-rag-uploads/{session_id}/`, **outside the git repo entirely** (not just gitignored — never inside the project directory at all, so there's no rule to maintain or forget).
- `mode="upload"` queries build their retriever from `sessions.index_dir(session_id)` — there is no code path from there back to the demo/research corpus.
- `mode="demo"` queries never touch session storage at all, even if the same browser has uploaded files.

**Verified for real, not just unit-tested** (see commands below): a request with no session cookie gets a clean 400 ("No receipts uploaded"), not another session's data. A same-session query in `demo` mode answered from the synthetic corpus only, correctly ignoring an uploaded receipt that WAS in that session.

### Verification — commands actually run, with real output

```bash
# PDF with embedded text (generated on the fly with PyMuPDF, not a real receipt scan)
curl -X POST http://127.0.0.1:8000/upload -F "file=@test_receipt.pdf;type=application/pdf"
# -> page_count: 1, chunks_indexed: 1, text correctly extracted, used_ocr: false

curl -X POST .../query -d '{"question": "What is the total at Test Coffee Shop?", "mode": "upload"}'
# -> "The total at Test Coffee Shop is $10.54." (correct), source: test_receipt.pdf (correct)

# session isolation
curl -X POST .../query -d '{"question": "...", "mode": "upload"}'   # NO cookie sent
# -> 400 "No receipts uploaded in this session yet."

curl -b cookies.txt -X POST .../query -d '{"question": "...", "mode": "demo"}'  # SAME session, demo mode
# -> answered from synthetic corpus only (demo_* chunk_ids), upload never leaked in

# missing information
curl -b cookies.txt -X POST .../query -d '{"question": "How many loyalty points were earned?", "mode": "upload"}'
# -> "I don't know.", has_sufficient_evidence: false

# validation
curl -X POST .../upload -F "file=@notes.txt;type=text/plain"      # -> 400 Unsupported file type
curl -X POST .../upload -F "file=@big.png"  # 11MB                 # -> 413 exceeds 10MB limit
curl -X POST .../upload -F "file=@receipt.png"  # tesseract absent  # -> 503 clean OCR-unavailable message

# cleanup
curl -b cookies.txt http://127.0.0.1:8000/session   # files: [test_receipt.pdf], total_chunks: 1
curl -b cookies.txt -X DELETE http://127.0.0.1:8000/session   # {"cleared": true}
curl -b cookies.txt http://127.0.0.1:8000/session   # files: [], NEW session_id issued
ls /tmp/doc-rag-uploads/{old_session_id}   # -> No such file or directory (confirmed physically deleted)

# full test suite
pytest tests/ -v
# -> 25 passed (9 new upload tests: valid PDF extraction, 4 invalid-file cases,
#    cited-answer query, no-upload error, session isolation, deletion)
```

### New dependencies

`pymupdf` (PDF text extraction + page rasterization), `pytesseract` (OCR wrapper), `python-multipart` (FastAPI file-upload support) — all free, local, pip-only. System `tesseract-ocr` package — free, local, requires `sudo apt-get install -y tesseract-ocr` (pending, see above).

### Known limitations

- **OCR accuracy unverified** (blocked on the pending tesseract install, see above).
- **No incremental indexing** — every new upload in a session rebuilds the whole session index from scratch. Fine at session scale (a handful of receipts), would not scale to many files per session.
- **Lazy expiry only** — sessions are swept on access, not by a background timer. A session that's never touched again after upload stays on disk until *something* triggers a sweep (the next request to this server, from anyone). Acceptable for a demo; a production deployment would want a periodic sweep task too.
- **CPU-only embedding for uploads** is a deliberate VRAM-safety tradeoff (see above), not a sign CPU is required — it's simply slower than the GPU-backed demo pipeline.

### Files changed/added this extension

New: `app/upload/` (5 files), `tests/test_upload.py`.
Modified: `app/main.py`, `app/rag_pipeline.py`, `app/schemas.py`, `app/config.py`, `.env.example`, `app/static/index.html`, `PROGRESS.md`.
Untouched: everything under `scripts/`, all research and demo corpus data, `PROJECT_GRAPH_REPORT.md`, `README.md`.
