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
