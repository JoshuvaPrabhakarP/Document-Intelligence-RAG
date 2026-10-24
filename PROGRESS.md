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

### A real blocker, resolved

This sandbox has no passwordless `sudo`, so I could not install the system `tesseract-ocr` package myself. **You ran the install yourself** (per your choice when asked) while everything else was built in parallel. Now resolved and fully re-verified:
- `tesseract --version` confirms **5.3.4** installed.
- Before testing through the app, ran tesseract directly on a synthetic test receipt image to confirm it reads cleanly in isolation.
- **Uploaded that image through the actual running API**: OCR ran (`used_ocr: true`), extracted text matched the raw tesseract output exactly ("OCR TEST DINER... TOTAL 19.00").
- **Asked a question requiring the OCR'd values** ("What is the total... and what did they order?") → correct answer ("$19.00 for a Burger, Fries, and Soda"), correct source citation (filename + matching excerpt), `has_sufficient_evidence: true`.
- PDF embedded-text extraction (no OCR needed) was already verified separately — most real PDFs have a text layer.
- The graceful-degradation path (clean 503 when OCR is unavailable) was verified earlier, before tesseract was installed, and remains in place as the fallback if OCR is ever unavailable again.

**The full upload → OCR → indexing → question → cited answer path is now genuinely verified end-to-end, not assumed.**

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

`pymupdf` (PDF text extraction + page rasterization), `pytesseract` (OCR wrapper), `python-multipart` (FastAPI file-upload support) — all free, local, pip-only. System `tesseract-ocr` 5.3.4 — free, local, now installed and verified.

### Known limitations

- **No incremental indexing** — every new upload in a session rebuilds the whole session index from scratch. Fine at session scale (a handful of receipts), would not scale to many files per session.
- **Lazy expiry only** — sessions are swept on access, not by a background timer. A session that's never touched again after upload stays on disk until *something* triggers a sweep (the next request to this server, from anyone). Acceptable for a demo; a production deployment would want a periodic sweep task too.
- **CPU-only embedding for uploads** is a deliberate VRAM-safety tradeoff (see above), not a sign CPU is required — it's simply slower than the GPU-backed demo pipeline.

### Files changed/added this extension

New: `app/upload/` (5 files), `tests/test_upload.py`.
Modified: `app/main.py`, `app/rag_pipeline.py`, `app/schemas.py`, `app/config.py`, `.env.example`, `app/static/index.html`, `PROGRESS.md`.
Untouched: everything under `scripts/`, all research and demo corpus data, `PROJECT_GRAPH_REPORT.md`, `README.md`.

## Milestone 2 extension — better source citations (complete)

Manual testing (with your own, non-synthetic document) surfaced a real gap: the "supporting excerpt" shown with each answer was just the first 200 characters of the retrieved chunk — for a long document, that's often an unrelated header, not the passage the answer actually came from. Reported case: a payment-method question should show the payment-options lines, not the top of the page.

### Fix

`app/excerpt.py` (new, ~45 lines) — scores each line of the chunk by lexical word-overlap against the generated answer (weighted higher) and the question (weighted lower, and the only signal available in the no-answer/error case), then returns a short window of context around the best-matching line instead of the first N characters. No extra embedding or LLM call — pure Python, no added latency or cost. Wired into `app/rag_pipeline.py::_format_sources`, which now takes `question`/`answer` instead of just the retrieved results.

**Why answer-weighted, not question-only:** the generated answer text usually repeats the exact extracted value (a dollar amount, a name, a payment method) almost verbatim from the source, which is a much more precise grounding signal than the question's generic wording alone.

### Verified — real document, not just the unit tests

Per your instruction, tested on a **synthetic** document only (fabricated clinic-visit-style receipt, explicitly labeled "NOT A REAL PATIENT," fake name/dates/amounts) — never touched, read, or referenced anything from your actual upload, which went directly from your browser to the session-isolated server and was never accessible to me in the first place.

```
Q: "What payment method was used?"
A: "Mastercard ending in 4477."
excerpt (NEW): "Payment methods accepted: Visa, Mastercard, Cash, HSA
                Paid via: Mastercard ending in 4477
                Amount paid: 200.00"
```
Old behavior would have shown the clinic's name/address block instead — confirmed via a unit test (`test_excerpt_centers_on_answer_relevant_line_not_document_start`) that explicitly asserts the facility header is NOT in the new excerpt.

**Insufficient-evidence still verified correct** on the same document: asked about information genuinely absent (date of birth, insurance policy number) → `"I don't know."`, `has_sufficient_evidence: false`.

5 new unit tests (`tests/test_excerpt.py`) + full suite re-run: **30/30 passing**.

### Known limitation

Lexical overlap only — no semantic matching. In the insufficient-evidence test above, the excerpt shown was the "Date of service" line (matched on the shared word "date" with "date of birth"), not a deeper match — a fair, explainable miss given the approach, not a bug, but worth knowing: a semantically-themed question with no exact keyword overlap could still pick an excerpt that's lexically-close-but-conceptually-off. Not pursued further here since it doesn't affect the actual answer/evidence-flag correctness, only which excerpt is displayed alongside a correctly-flagged "I don't know."

## Milestone 2 extension — polished portfolio-grade interface (complete)

Full visual/UX redesign of `app/static/`, entirely frontend-scoped — **zero backend files changed**. Every requirement (drag-and-drop, document cards, suggested-question chips, a "Technical details" disclosure, even page references in citations) turned out to be achievable from data the API already returns, so nothing in `app/main.py`, `rag_pipeline.py`, or any other backend file needed to move.

### What changed

```
app/static/
├── index.html   # semantic markup only (was: markup + inline CSS + inline JS)
├── style.css    # NEW — dashboard theme: navy/blue accents, light surfaces, responsive
└── app.js       # NEW — all interaction logic, extracted from inline <script>
```

- **Layout:** clean dashboard — `Document Intelligence` heading, the exact requested subtitle, two ARIA-proper tabs (`Try Sample Receipts` / `Upload Your Documents`), consistent card/spacing system, responsive down to mobile widths (tested via CSS media query, not a live device).
- **Upload:** the plain `<input type="file">` is now a drag-and-drop zone (mouse, keyboard — Enter/Space, and click-to-browse all work), auto-uploads on drop/select, shows one honest "Processing — extracting text and indexing…" status (no fake percentage, since the server genuinely does this as a single blocking step — a true multi-stage progress bar would need a backend streaming change, which is out of scope for an interface-only pass and is noted here rather than faked).
- **Document cards:** compact — filename, page count, extracted character count, processing warning if any, plus the existing expandable full-extracted-text panel. Images get a visual badge, PDFs get a distinct badge ("preview where supported" — true client-side PDF rendering would need pdf.js, which the "avoid a framework unless necessary" instruction argued against pulling in for this).
- **"Clear documents"** is a small `.btn-ghost` button (not a dominant red CTA like the previous version), with a native `confirm()` prompt before it actually deletes anything.
- **Chips:** demo-mode chips are still the real, answerable sample questions from `/sample-questions` (unchanged, curated). Upload-mode chips are a new, deliberately **generic** list ("What is the total amount?", "Who is the vendor?", etc.) — not tied to any specific fabricated answer, since they have to work against whatever the user actually uploads.
- **Answer states:** success/insufficient-evidence/error are now visually AND textually distinct (a checkmark+"Answered", a warning triangle+"Insufficient evidence", an X+"Error" — icon and label together, never color alone).
- **Citations:** filename plus a parsed page number (`chunk_id` already encodes `_p{N}` for uploads — parsed client-side in `app.js`, no backend change) beneath the answer.
- **Technical details:** chunk IDs, raw scores, and all latency numbers moved into a collapsed `<details>` per answer instead of always being visible.
- **State:** switching tabs now restores that mode's last answer instead of wiping it; clearing documents explicitly wipes the upload mode's last answer (since its citations would point at now-deleted documents) — demo mode's last answer is untouched by a clear.
- **Safety:** every dynamic value (answers, excerpts, filenames, error messages, extracted text) is inserted via `textContent`/`createElement`, never string-concatenated into `innerHTML` — the previous version already escaped correctly, but this rewrite removes the `innerHTML`-with-manual-escaping pattern entirely in favor of DOM construction, which can't be bypassed by a missed escape.
- **Accessibility:** proper `role="tab"`/`aria-selected`/`aria-controls` on the tabs, `aria-live="polite"` on the status banner/file list/result area so screen readers announce updates, a real `<label>` for the question field, visible `:focus-visible` rings sized for keyboard navigation, `prefers-reduced-motion` respected for the spinner.
- **Copy:** removed the one direct "see PROGRESS.md" reference from the UI; demo/retention/privacy language rewritten for an end user, kept factually identical to before (synthetic data, local-only OCR, 30-minute auto-expiry).

### Verified — functionally, via the real running server

No browser automation tool is available in this sandbox (checked: no Playwright, Selenium, or Chrome installed) — **visual rendering at desktop/mobile widths has not been visually inspected and remains pending your review.** Everything reachable without a browser was verified for real against the live server:
- All three static files (`index.html`, `style.css`, `app.js`) load with correct content-types.
- `app.js` passes `node --check` (syntax-valid) and every DOM ID it references was cross-checked to exist in `index.html`.
- Demo-mode query end-to-end: correct answer, correct source.
- Multi-page PDF upload + query: correct answer drawn from page 2 specifically, and the citation's `chunk_id` (`{file_id}_p2`) parses to "page 2" exactly as the new citation UI expects.
- Session listing shape matches what the new file-card renderer consumes field-for-field.
- The "no documents uploaded, query upload mode anyway" error path returns the exact `detail` field the new error-card renderer reads.
- Full backend test suite re-run: **30/30 still passing** (expected — no backend file was touched).

### Known limitations

- **No live visual verification** (see above) — please check desktop and mobile rendering yourself and report anything that looks off.
- **Upload progress is single-stage, not multi-stage** — accurately described as one step because that's what's actually happening server-side right now; true per-stage (upload → OCR → indexing) progress would need a streaming/polling backend change, out of scope for this pass.
- **PDF preview is a badge, not a rendered thumbnail** — avoided pulling in pdf.js per the "no framework unless necessary" instruction. Image uploads could get a real thumbnail (via `URL.createObjectURL`) in a future pass if wanted.

### Files changed/added this extension

New: `app/static/style.css`, `app/static/app.js`.
Rewritten: `app/static/index.html` (markup only now).
Modified: `PROGRESS.md`.
Untouched: every backend file — `app/main.py`, `app/rag_pipeline.py`, `app/schemas.py`, `app/config.py`, `app/upload/`, all of `scripts/`, all research/demo data.

## Milestone 2 extension — colorful portfolio redesign (complete)

The previous pass was a clean but visually plain dashboard; this one is a substantial visual redesign per explicit direction — again entirely frontend-scoped, zero backend files touched.

### What changed

- **Color system** (new CSS custom properties): navy `#0b1226`→violet `#5b21b6` gradient header, pale lavender page background (`#f4f2fc`) with two large blurred radial-gradient "blobs" (blue + violet, 22% opacity) fixed behind the content so the page never reads as flat — explicit requirement.
- **Layout:** two-column CSS grid on desktop (`minmax(300px,380px) 1fr`) — left column holds mode selection (segmented tabs, demo/upload panels), right column holds the question composer and answer, both inside `.card` surfaces (white, 18px radius, soft shadow). Collapses to a single stacked column under 860px.
- **Segmented tabs:** filled pill control on a lavender track; the active tab gets the blue→violet gradient fill and a matching shadow, replacing the previous underline-style tabs.
- **Header:** brand icon (document SVG in a translucent tile) + "Document Intelligence" + the requested tagline "Answers grounded in your documents.", white text on the gradient.
- **Upload dropzone:** lavender-tinted, violet dashed border, a gradient circular upload-icon badge; drag-over state switches to a teal/success tint with a slight scale transform.
- **Chips:** now carry a small icon alongside the text. Demo chips (fetched from `/sample-questions`, unchanged data) get an icon picked client-side by keyword match (total→$, vendor→store, date→calendar, payment→card, items→list, fallback→"?"). Upload chips are still the generic, non-answer-specific list from the previous pass, now with the same icon treatment.
- **Answer card:** status row (icon + bold label: "Answered" / "Insufficient evidence" / "Error") stays from the prior pass; sources heading is now teal, each source card has a teal left-accent border and a violet filename/page label. Technical details stay in the same collapsed `<details>`, now on a tinted background for contrast.
- **Empty state:** before any question is asked (or right after clearing documents), the right column now shows a document-with-magnifying-glass illustration and "No answer yet — pick a suggested question or type your own…" instead of being visually blank. This is new: the prior version just left the area empty.
- **"Ask document" button:** blue→violet gradient, hover lift, icon + label — the label lives in its own `<span id="ask-btn-label">` now specifically so the in-flight "Asking…" state swaps only the text, never destroying the icon markup around it.

### What did NOT change

No fake data was added anywhere in this pass, per explicit instruction: no invented confidence percentages, no multi-stage progress simulation (upload status is still the same single honest message as before, just restyled), no decorative controls that don't do anything. Every element that looks clickable is wired to real existing behavior.

### Verified — functionally, via the real running server

**Visual rendering has NOT been inspected** — this sandbox still has no browser automation tool (no Playwright/Selenium/Chrome), so desktop/mobile appearance is reported as pending your review, not assumed to look right. Everything reachable without a browser was re-verified:
- All three static files load with correct content-types; `app.js` passes `node --check`; every DOM ID it references (including the new `ask-btn-label`) was cross-checked against the new `index.html`.
- Demo-mode query: correct answer, correct evidence flag.
- Full upload→query→citation→clear flow on a fresh synthetic 2-page PDF: correct answer ("Visa"), correct page attribution (page 2, where the payment lines actually are), correct excerpt (the payment-method lines specifically, confirming the earlier excerpt-selection fix still works through the new UI), clear succeeds.
- Full backend test suite re-run: **30/30 passing** (unchanged, as expected — no backend file was touched).

### Files changed this extension

Rewritten: `app/static/style.css`, `app/static/index.html`, `app/static/app.js` (extended, not replaced — all prior logic for uploads/session isolation/safe rendering is intact, only chip-icon selection, the empty-state renderer, and the button-label element reference were added/changed).
Modified: `PROGRESS.md`.
Untouched: every backend file, exactly as the prior extension.
