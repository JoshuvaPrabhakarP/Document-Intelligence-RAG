# Enterprise Document Intelligence RAG — GRAPH REPORT
> Living status document. Update this whenever a decision is made, a gate is passed, or scope changes — treat it the way `GRAPH_REPORT.md` was treated in the extraction project: the canonical source of truth, not the chat history.

**This is a fresh, standalone project with its own new GitHub repo — not a continuation phase of `hybrid-doc-extractor`.** It imports specific reusable assets (dataset, trained Transformer checkpoint, inference-loading code) from that prior project, but has its own independent phase numbering starting at Phase 1.

**Prior project (asset source only):** `github.com/JoshuvaPrabhakarP/hybrid-doc-extractor` (its Phases 1-8, complete)
**This project:** repo created and live — `github.com/JoshuvaPrabhakarP/Document-Intelligence-RAG`
**Current stage:** This project's Phase 1, Step 0 — COMPLETE. Ready to begin Step 1.

---

## 1. Research questions (locked)

1. Does structured metadata extracted from business documents improve RAG retrieval and answer quality?
2. How much does imperfect (model-extracted) metadata erode that benefit vs. ground-truth metadata?
3. Which retrieval strategy (dense, lexical, hybrid, metadata-aware) offers the best trade-off between retrieval quality, latency, and robustness across business document types?

**Novel contribution:** the metadata-quality linkage table — correlating upstream extraction F1 (per field, per architecture: Transformer/Hybrid/Mamba) against downstream Recall@10. This is what makes the extraction project and this project one coherent two-part story rather than two unrelated pieces.

---

## 2. The nine-step build plan

| Step | Name | Status |
|---|---|---|
| 0 | Environment & data prep | ✅ Complete |
| 1 | Vector-only baseline + structured logging | 🔶 Ready to start |
| 2 | BM25 + hybrid fusion | ⏳ Not started |
| 3 | Dual-track metadata layer (GT / Transformer) | ⏳ Not started |
| 4 | Core 5×2 eval matrix (objective + Ragas) | ⏳ Not started |
| 5 | Reranker ablation (quarantined, best-config only) | ⏳ Not started |
| 6 | Error analysis (from logs, no re-running) | ⏳ Not started |
| 7 | README + write-up | ⏳ Not started |
| 8 | Deployment (FastAPI → Gradio → Cloudflare Tunnel) | ⏳ Not started |

---

## 3. Data source & reading-order reconstruction — RESOLVED

### The problem discovered during pre-flight
`data/processed/{train,val,test}.jsonl` (the extraction project's BIO-tagged files) are **not usable for chunking directly** — unspaced, non-reading-order OCR boxes (see original diagnosis below, unchanged).

### The fix
`data/raw/sroie/SROIE2019/{train,test}/{img,box,entities}` and `data/raw/wildreceipt/wildreceipt/{train,test}.txt` retain per-box polygon coordinates, allowing genuine reading-order reconstruction. Script: `scripts/reconstruct_reading_order.py`.

**Status: COMPLETE.** `data/rag_corpus/{wildreceipt,sroie,cord}.jsonl` written — see final totals below.

### Three real bugs found and fixed during verification (all via spot-checking actual output, not trusting aggregate match-rate stats)

**Bug 1 — spurious short-string matches.** Original fuzzy matcher's `a in b or b in a` substring shortcut let single characters (e.g. `"1"`) trivially match inside longer date strings (`"12-01-19"`), winning a false perfect score and getting tagged `B-DATE` instead of the real date. Fixed with a minimum-length guard (`len(shorter) >= 4`) before applying the substring shortcut.

**Bug 2 — tag spillover onto unrelated adjacent words.** SROIE box files sometimes group a whole receipt line into one box (date + time + register code all as one string). Whole-box tagging meant matching the date anywhere in the box tagged the *entire* box, sweeping unrelated codes (`SH01`, `ZK09`) and even label words (`DATE:`) into the entity span. Fixed by switching date/total matching to a sliding word-count-exact window, tagging only the words that actually correspond to the entity value.

**Bug 3 — company matching regressed to 92.1%, then over-matched into an unrelated adjacent box.** Company names (a) are tokenized differently between `entities.txt` and the OCR box text, and (b) can span multiple consecutive boxes — neither solvable by a fixed single-box window. Fixed with whitespace-normalized matching across sliding 1-3-box windows — which then surfaced a *fourth* issue: a recurring artifact box (`"TAN WOON YANN"` / `"TAN CHAY YEE"`, present as the first line of nearly every SROIE receipt, unrelated to the actual company) was getting swept into the match via a tied score that a first-found tie-break resolved incorrectly. Fixed by preferring the shortest window on tied/near-tied scores.

**Verification method used throughout:** 8-document eyeball spot-check (mirrors original Phase 2's 3-example spot check) plus two corpus-wide scans — one for suspiciously short `B-DATE` tokens, one for the most common `B-NAME` first-words (would surface a repeated artifact bleeding into every document, which is exactly what Bug 3 was). Final corpus-wide scan came back clean: top 15 `B-NAME` first words are all legitimate business-name fragments, no repeated artifact.

### Verified final SROIE match rates
- company: 973/973 (100.0%)
- date: 972/973 (99.9%)
- total: 971/973 (99.8%)

### Open item
~~CORD not included.~~ **RESOLVED.** CORD was still cached locally (`~/.cache/huggingface/datasets/naver-clova-ix___cord-v2`), loaded offline via the `datasets` library (`HF_HUB_OFFLINE=1`) — no re-download needed. Since `cord-v2` has no per-box coordinates (Donut-style structured JSON only, confirmed against the original Phase 2 log's finding), it uses a simpler dedicated path: readable text built directly from the structured `menu`/`total` fields in their natural list order, with only the `total_price` value tagged `B-AMOUNT`/`I-AMOUNT` — no fuzzy matching needed since the text is generated, not reconstructed. `NAME`/`DATE` intentionally stay `O` for all CORD documents, matching the field's genuine absence in the source data (not a limitation to fix).

### Final corpus totals
- WildReceipt: 1,739 documents
- SROIE: 973 documents
- CORD: 1,000 documents
- **Total: 3,712 documents** (vs. original project's 3,684 — the 28-document difference is CORD-only; original `preprocess.py` likely applied an additional filter, e.g. dropping records with missing `total_price`, that this loader doesn't replicate. Not a data-quality issue, just a minor scope difference worth noting.)

Output files: `data/rag_corpus/{wildreceipt,sroie,cord}.jsonl` (gitignored, regenerable via `scripts/reconstruct_reading_order.py`).

---

## 4. Model checkpoints — RESOLVED

- All 18 runs present: `checkpoints/{transformer,mamba,hybrid_5:1,hybrid_4:2,hybrid_3:3,hybrid_2:4}` × seeds {42,123,456}, each `best.pt` (~79MB) + `training_log.json`
- Best Transformer: **seed456, test entity-F1 0.6303**, CRF-enabled (`use_crf: True`), matches "Round 3" from the extraction project
- `best.pt` = raw model `state_dict` (no optimizer state), CRF layer included since it's part of the model itself
- Loading code: `app_gradio/inference.py::load_model()` is **directly reusable** — builds from `configs/experiment.yaml` + `load_checkpoint()`, just needs `PROJECT_ROOT` adjusted if copied into the new repo
- Only the Transformer checkpoint is needed to start (Step 1-4). Mamba/Hybrid checkpoints needed later, only for the linkage table (Section 6 below) — gated on the gcc issue (Section 5).

---

## 5. Environment — mostly resolved, one open risk

**Architecture decision (locked):** Two fully separate venvs, communicating only via files on disk — no shared dependencies, no risk of the fragile Mamba/CUDA stack being destabilized by new installs.

- **Old venv** (`~/hybrid-doc-extractor/venv`, untouched): extraction/checkpoint inference only. Stays pinned to `transformers==4.44.2`, `tokenizers==0.19.1`, `torch==2.7.0+cu128`, gcc-11 toolchain. Its only job going forward: run the Transformer checkpoint over the corpus when needed and dump predictions to JSON/JSONL.
- **New venv** (`~/document-intelligence-rag/venv`): all RAG work — chunking, indexing, retrieval, evaluation, generation. Installs its own `torch` transitively via `sentence-transformers`, whatever version it wants — no coordination needed with the old venv's pins.
- **Handoff mechanic:** old venv writes plain files (JSON/JSONL), new venv reads them. No live process communication between the two.

| Component | State |
|---|---|
| GPU | RTX 5060 Laptop, 8GB VRAM, confirmed working |
| CUDA | 12.8, `torch==2.7.0+cu128` (old venv only), `cuda.is_available()` = True |
| gcc | **13.3.0** — extraction project's install notes specified gcc-11 for Mamba's CUDA kernels. Not yet re-verified. Only matters for the old venv / Mamba-Hybrid checkpoints. |
| New venv | Not yet created — commands drafted, not yet run |
| Ollama / vLLM / llama.cpp | Not installed — system-level install, shared across venvs (not a pip package) |
| FAISS / sentence-transformers / rank_bm25 / llama-index / ragas / FlagEmbedding | Not installed — new venv install list finalized (Section 6) |
| Local LLM choice | **RESOLVED: Qwen 3 (8B), via Ollama** |
| VRAM budget / LLM placement | **RESOLVED: CPU via Ollama.** Extractor + BGE-large stay on GPU (small, run frequently during indexing/queries); LLM runs on CPU since generation only happens during the eval/answer step, not every indexing iteration — accepts slower generation in exchange for zero VRAM contention and simpler setup (no quantize/swap logic needed). Ollama installed system-wide, shared across both venvs (not a pip package). |

**Resolved:** `import mamba_ssm` smoke test passed in the old venv — import succeeded AND a real GPU forward pass produced the correct output shape `(1, 32, 64)`, confirming the compiled `sm_120` kernels genuinely work despite the gcc-13 (vs. original gcc-11) system compiler. This makes sense in hindsight: the compiled `.so` files were already built against gcc-11 when the extraction project set this up — the current system gcc only matters if `mamba_ssm` needed rebuilding from source again, which it doesn't. **Mamba and Hybrid checkpoints are usable now, no deferral needed for the linkage table step.**

---

## 6. Step 0 install list — COMPLETE

Ran inside the **new venv only** (`~/document-intelligence-rag/venv`). Old venv was not touched.

| Package | Purpose | Confirmed version |
|---|---|---|
| `faiss-cpu` | Vector index for dense retrieval | 1.14.3 |
| `rank_bm25` | Lexical/keyword retrieval | installed |
| `sentence-transformers` | Runs `bge-large-en-v1.5` for chunk embeddings | 5.6.0 |
| `llama-index` | Orchestration — chunking, metadata-filtered retrieval | importable |
| `ragas` | LLM-judged eval track | **0.2.15** (downgraded from 0.4.3 — see fix below) |
| `FlagEmbedding` | Runs `bge-reranker-large` — Step 5 only | importable |
| `torch` | Pulled in transitively by sentence-transformers | **2.7.0+cu128** (fixed — see below) |

**Not yet installed:** FastAPI/Gradio/Cloudflare Tunnel — Step 8, deliberately last.

### Real issues hit and fixed during install

**1. `ragas` import failure — upstream bug, not local misconfiguration.**
`ragas` (0.3.9 and 0.4.3 both) imports `ChatVertexAI` from `langchain_community.chat_models.vertexai` — a path that was removed from `langchain-community` in favor of a separate `langchain-google-vertexai` package. `ragas` never pinned a compatible `langchain-community` version, so pip resolved to the newest one, which no longer has that module. Confirmed as an open upstream bug (`vibrantlabsai/ragas#2745`).
- Attempted fix: pin `ragas==0.2.15` + `langchain-community==0.2.19` + `langchain>=0.2.17,<0.3.0` (the bracket `langchain-community==0.2.19` itself requires) — reduces but doesn't fully eliminate resolver risk from other packages pulling newer `langchain-community`.
- **Final fix applied:** manual local shim module at
  `venv/lib/python3.12/site-packages/langchain_community/chat_models/vertexai.py`,
  re-exporting `ChatVertexAI` from `langchain_google_vertexai` with a graceful
  fallback. **Not tracked in git** (lives inside `venv/`, which is gitignored) —
  if the venv is ever rebuilt from `requirements.txt` alone, this file must be
  recreated manually. Documented in `ENVIRONMENT_NOTES.md`.
- Confirmed working: `ragas 0.2.15` imports cleanly.

**2. `torch` pulled the wrong CUDA build.**
`sentence-transformers` transitively installed `torch==2.12.1+cu130` — built against a newer CUDA than the installed driver (12.8) supports, so `torch.cuda.is_available()` returned `False` silently (no hard error, just silent CPU fallback — same "trust but verify" trap as the extraction project's Blackwell kernel issue).
- **Fix:** `pip uninstall torch torchvision torchaudio`, reinstall via
  `--index-url https://download.pytorch.org/whl/cu128`, matching the driver.
- Confirmed working: `torch 2.7.0+cu128`, `cuda.is_available() = True`, GPU detected (RTX 5060 Laptop GPU).

**Full working stack recorded in `requirements.txt` (via `pip freeze`) and `ENVIRONMENT_NOTES.md`, both committed to the new repo.**

---

## 7. Untriaged files — open question

**RESOLVED.** Findings from inspection (`diff -rq app/ app_gradio/`): `app/` is the original Streamlit dashboard draft (has `dashboard.py`), deliberately superseded by the Gradio rebuild in `app_gradio/` (has `app.py` + `theme.py` instead) — matches the session log's documented rebuild. Neither dashboard is needed for this project; only `app_gradio/inference.py::load_model()` matters (already confirmed reusable, Section 4).

Actions taken:
- **Deleted:** `checkpoints_phase5.zip`, `evaluation_phase6.zip` (redundant transfer backups, ~1.1GB freed), `cloudflared-linux-amd64.deb` (leftover installer)
- **Kept as-is:** `app/` (harmless, low priority, not needed but not worth the cleanup effort right now)
- **Committed to the old repo:** `evaluation/` — real, completed Phase 6 work (`evaluate.py`, `benchmark.py`, `seqlen_ablation.py`, `significance_test.py` + result JSONs) that had never been committed, meaning it previously had zero backup beyond the one machine. Now pushed.

---

## 8. The 5×2 experimental matrix (design reference)

| System | Dense | BM25 | Metadata | Purpose |
|---|---|---|---|---|
| Vector-only | ✓ | ✗ | ✗ | Baseline |
| BM25-only | ✗ | ✓ | ✗ | Lexical baseline |
| Hybrid | ✓ | ✓ | ✗ | Modern retrieval baseline |
| Metadata + Vector | ✓ | ✗ | ✓ (GT / Transformer) | Does structure help? |
| Metadata + Hybrid | ✓ | ✓ | ✓ (GT / Transformer) | Full system |

Every metadata-aware row runs twice (GT and Transformer) — 5 systems × 2 metadata sources = the core matrix.

**Metadata-quality linkage table (target shape, to fill in once Step 3/4 run):**

| Metadata Source | Vendor F1 | Date F1 | Amount F1 | Retrieval Recall@10 |
|---|---|---|---|---|
| Ground Truth | 100% | 100% | 100% | — |
| Transformer | — | — | — | — |
| Hybrid | — | — | — | — |
| Mamba | — | — | — | — |

## 9. Gating checklist for Step 0 — ALL COMPLETE

- [x] New venv created (`~/document-intelligence-rag/venv`, Python 3.12)
- [x] Step 0 stack installed in new venv (Section 6) — two real fixes applied: `ragas`/`langchain-community` shim, `torch` CUDA build pin
- [x] New repo created on GitHub and pushed (`Document-Intelligence-RAG`)
- [x] Reading-order reconstruction script written + verified (8-doc spot-check + 2 corpus-wide scans, 3 real bugs found and fixed)
- [x] BIO tags re-associated with reconstructed text — **done as part of reconstruction itself** (tags are generated directly from raw sources, not re-associated after the fact — see Section 3)
- [x] CORD integration — loaded offline from existing HF cache, structured-JSON path (no fuzzy matching needed), 1,000 documents
- [x] `import mamba_ssm` smoke test in **old venv** on gcc-13 toolchain — PASSED (import + real GPU forward pass both succeeded)
- [x] LLM placement decision made — CPU via Ollama, Qwen 3 (8B)
- [x] `app/` triage — resolved: zips + .deb deleted, real evaluation/ work committed to old repo, app/ left in place (harmless, superseded by app_gradio/)

---

## 10. Next immediate action (as of this update)

**Step 0 is fully complete.** All eight gating items resolved: venv, dependency stack, GitHub repo, reading-order reconstruction (3 sources, 3,712 documents), mamba_ssm smoke test (passed), LLM placement (CPU via Ollama, Qwen 3 8B), and app/ triage.

Next: **Step 3 — Dual-track metadata layer** (Steps 1 and 2 now complete — see Sections 11-12).

---

## 11. Step 1 progress — Vector-only baseline + structured logging

### 1a. Chunking — COMPLETE

Script: `scripts/chunk_corpus.py`. One chunk per document (receipts are short, self-contained — no internal splitting needed).

**Result:** 3,712 chunks written to `data/chunks/chunks.jsonl`, unifying all three sources with stable `chunk_id`s and clean ground-truth field extraction (`vendor`/`date`/`amount` as reconstructed strings, not just raw BIO tags) — this is what Step 3's dual-track metadata layer will consume directly.

**Ground-truth field coverage:**
- amount: 3,595/3,712 (96.8%)
- vendor: 2,281/3,712 (61.4% overall; ~84% of WildReceipt+SROIE docs — CORD never has this field by design, dragging the overall percentage down)
- date: 2,442/3,712 (65.8% overall; ~90% of WildReceipt+SROIE docs, same CORD effect)

Missing values are genuine source-data limitations (OCR/labeling gaps in the original WildReceipt/SROIE datasets), not artifacts of the reconstruction pipeline — confirmed via the corpus-wide coverage stats built into the script.

Chunk length: 14–1,858 chars, avg 371. Five chunks under 20 chars flagged and inspected — all genuine minimal single-item CORD receipts, not corruption.

**Bug found and fixed during this step:** the coverage-stats block crashed with `ZeroDivisionError` when run against an empty/missing corpus (wrong working directory on first attempt). Fixed with an explicit empty-check that fails with a clear diagnostic message instead of a raw traceback.

### 1b. Embedding + FAISS index — COMPLETE

Script: `scripts/embed_index.py`. Model: `BAAI/bge-large-en-v1.5` on GPU, `IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity).

**Result:** 3,712 vectors, dim 1024, embedded in 155.4s (~24 chunks/sec on the RTX 5060). Index + parallel metadata (`chunk_id`/`doc_id`/`source`/`text`/`ground_truth` per vector position) + config (model name, similarity method, query prefix) saved to `data/index/` — gitignored, fully regenerable from `chunks.jsonl`.

**Design note carried forward for the retrieval script:** BGE is asymmetric — passages (corpus chunks) get no prefix, but queries need `"Represent this sentence for searching relevant passages: "` prepended at retrieval time. Saved explicitly in `index_config.json` so this isn't re-discovered or gotten wrong later.

**Sanity check passed:** query *"What is the total amount?"* returned three receipts genuinely containing totals, scores 0.66–0.68 (a healthy cosine similarity band, not suspiciously perfect or near-zero) — real evidence the embedding space is behaving sensibly, not just that the code ran without error.

### 1c. Structured logging schema — COMPLETE

Script: `scripts/logging_schema.py`. `QueryLogger` class, one instance per system-run, writing to `logs/{system}__{metadata_source}.jsonl` (self-describing filename, no need to open a file to know what's in it).

**Schema fields** (per query): `query_id`, `query_text`, `question_type` (A-E taxonomy), `retrieved_chunks` (chunk_id/score/text), `metadata_filters_applied`, `reranker_scores`, `generated_answer`, `ground_truth_answer`, `retrieval_latency_ms`, `generation_latency_ms`, plus `extra` as an escape hatch for system-specific detail (e.g. hybrid fusion weights) without forcing every system into an identical schema.

**Design decisions:**
- Scores are NOT normalized across systems in the log — a `vector_only` cosine score and a `bm25_only` score are genuinely different numbers; any cross-system normalization is an explicit analysis-time decision (Step 6), not silently baked in here.
- `system`/`metadata_source`/`question_type` are validated against fixed vocabularies — typos become loud errors immediately instead of silently fragmenting logs.
- `reset()` is explicit, never automatic — re-running a script can't silently wipe prior logged queries.

**Verified:** self-test (write → read back → assert round-trip integrity) passes both in isolated testing and on the target machine before being trusted as foundational infrastructure for every later step.

---

## STEP 1 — COMPLETE (chunking, embedding+FAISS, logging schema all done)

---

## 12. Step 2 progress — BM25 + hybrid fusion

Scripts: `scripts/dense_retrieve.py` (extracted from embed_index.py's inline check into a proper reusable `DenseRetriever` class), `scripts/bm25_retrieve.py`, `scripts/fusion.py` (Reciprocal Rank Fusion, k=60 standard constant).

**Fusion method:** RRF chosen over weighted score fusion — combines by rank position, not raw score value, avoiding the need to normalize cosine similarity against BM25 scores (genuinely different scales, no principled way to combine them as raw numbers).

**Real bug found and fixed:** initial BM25 results were completely wrong-topic — querying *"What is the total amount on the Walmart receipt?"* returned unrelated BHPETROL gas station receipts as the top BM25 matches, with zero Walmart connection. Root cause, confirmed via diagnostic: BM25 tokenization had no stopword removal, so common words in the query (`is`, `the`, `on`, `what`) were treated as real search terms. A BHPETROL chunk happened to contain a boilerplate disclaimer — *"This is not a receipt, a tax invoice will be issued..."* — whose natural-language phrasing shared more stopwords with the query than an actual Walmart receipt (which has no reason to contain conversational words like "is" or "on"). Fixed by adding stopword removal to `tokenize()` — unlike dense embeddings (which use function words as context), BM25 needs this explicitly since it treats every query token as an equally weighted independent signal.

**Verified after fix:** re-ran the same query — BM25's top results are now genuinely Walmart-related, and hybrid fusion shows `found_by=dense+bm25` (both retrievers agreeing) on every top-5 result, versus completely disjoint lists before the fix.

**Observation for later (not a bug):** hybrid top results currently lean toward Walmart promotional/boilerplate text over chunks with actual dollar totals — RRF correctly rewarding chunks strong in both lists, but "Walmart" as a keyword dominates over "total amount" as a concept. Worth watching in Step 4's Recall@k evaluation and Step 6's error analysis, not something to chase further at this stage.

**Additional verification queries run:**
- *"WALMART receipt"* — clean confirmation the stopword fix generalizes: dense, BM25, and hybrid all returned genuinely Walmart-related results, full `dense+bm25` agreement across the top 5.
- *"receipt for refueling a car"* — surfaced a real, legitimate finding rather than a bug: dense retrieval correctly identified genuine gas-station/fuel-pump receipts (Shell, `FUELKINGSERVICECENTER`, litres/price-per-litre fields) with zero literal keyword overlap — real semantic understanding. BM25, by contrast, returned car WASHES almost exclusively, because those chunks contain a transaction field literally labeled `"Car#304"` (the `#` breaks tokenization into the literal token `"car"`), which lexically matches the query's `"car"` despite being semantically unrelated (a car wash transaction ID, not the vehicle-refueling concept). **Consequence:** in the hybrid fusion for this query, BM25's confidently-wrong car-wash matches crowded out dense's correct fuel-station results — hybrid was measurably worse than dense-only here. Logged as a genuine research finding (directly relevant to RQ3: retrieval strategy trade-offs), not something to engineer around at this stage — exactly the pattern Step 6's "BM25-specific failures" error category exists to capture systematically.

---

## 13. Step 3 progress — Dual-track metadata layer

### 3a. Tokens added back to chunks.jsonl — COMPLETE

Small addition to `chunk_corpus.py`: chunk records now carry `tokens` alongside `text`, for potential future alignment checks. `chunk_id`s unchanged (same ordering), so no downstream rebuild needed.

### 3b. Transformer-extracted metadata — COMPLETE

Script: `extract_metadata_transformer.py`, run in the **old venv**, importing `app_gradio.inference.extract_entities_from_text()` / `entities_to_field_dict()` directly — real inference code, not reimplemented. Writes output into the *new* repo's `data/metadata/transformer_extracted.jsonl` — a genuine cross-venv file-handoff (plain OS write, no shared imports).

**Result:** all 3,712 chunks processed in 57.8s (~64 chunks/sec) using the Transformer checkpoint, seed 456 (best test F1, 0.6303, per Section 4).

**Predicted field coverage vs. ground truth:**
| Field | Ground truth | Transformer-predicted |
|---|---|---|
| vendor | 61.4% | 68.3% (over-predicts) |
| date | 65.8% | 64.0% |
| amount | 96.8% | 81.4% (under-predicts) |

**Sample-level verification confirms genuine, expected model error** (not a pipeline bug): real false positives (a garbled OCR line mistagged as a second vendor name; a barcode number mistagged as an amount; a marketing tagline "Our guac rocks." mistagged as a business name where ground truth has no vendor at all), alongside correct extractions where the model got it right. This mix is exactly what "imperfect real-world extraction" should look like, and is the actual signal RQ2 needs.

### 3c. Per-field extraction accuracy (precision/recall) — COMPLETE

Script: `scripts/eval_metadata_extraction.py`.

**Real methodology bug found and fixed before trusting results:** the first version compared ground truth's single first-matched value against the Transformer prediction's *entire comma-joined string of all matches* (since `app_gradio.inference.entities_to_field_dict()` joins multiple spans for display purposes). This is an apples-to-oranges comparison — receipts commonly have multiple genuine AMOUNT spans (subtotal, tax, total), so comparing "one value" against "a joined list of values" systematically failed even when the model correctly found the right value among several. Symptom: AMOUNT's exact-match F1 was a nonsensical 0.153 while fuzzy-match F1 was 0.661 — a huge, suspicious gap.

**Fix:** `extract_metadata_transformer.py` now outputs proper `all_vendors`/`all_dates`/`all_amounts` lists (mirroring `chunk_corpus.py`'s ground-truth schema exactly) instead of relying on the joined-string display format. `eval_metadata_extraction.py` rewritten to do genuine set-based comparison — matching each GT value against the full set of predicted values and vice versa, not a single-value-vs-string comparison.

**Verified fix worked:** AMOUNT's exact/fuzzy F1 converged from a suspicious 0.153/0.661 gap to a coherent 0.591/0.613 — exactly what's expected once the comparison stopped systematically penalizing correct multi-value extractions.

**Final per-field accuracy (Transformer, seed 456, vs. ground truth):**

| Field | Exact P / R / F1 | Fuzzy P / R / F1 |
|---|---|---|
| vendor | 0.803 / 0.915 / 0.855 | 0.823 / 0.937 / 0.876 |
| date | 0.619 / 0.650 / 0.634 | 0.887 / 0.930 / 0.908 |
| amount | 0.530 / 0.667 / 0.591 | 0.550 / 0.692 / 0.613 |

**Sample-level verification confirms genuine, interpretable model behavior:** e.g. `chunk_000005` — model correctly extracted subtotal and tax but missed the actual total amount, a real and meaningful recall failure on the most important amount field. This is the honest "how much does imperfect extraction erode the metadata benefit" signal RQ2 needs — now on solid methodological footing.

Full report (including mismatch samples per field) saved to `data/metadata/extraction_accuracy_report.json` (gitignored, regenerable).

### 3d. Wire metadata filters into retrieval — NEXT

---