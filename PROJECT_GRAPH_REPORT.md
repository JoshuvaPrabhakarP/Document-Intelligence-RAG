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
| 4 | Core 5×2 eval matrix (objective + Ragas) | ✅ Complete |
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

**Metadata-quality linkage table — FILLED IN (Step 4, see Section 14 for full methodology):**

| Metadata Source | Vendor F1 (fuzzy) | Date F1 (fuzzy) | Amount F1 (fuzzy) | Retrieval Recall@10 (filtered queries) |
|---|---|---|---|---|
| Ground Truth | 100% | 100% | 100% | 76.4% |
| Transformer | 87.6% | 90.8% | 61.3% | 76.4% |
| Hybrid | — | — | — | not yet run (needs Mamba/Hybrid checkpoints over the corpus) |
| Mamba | — | — | — | not yet run (needs Mamba/Hybrid checkpoints over the corpus) |

Recall@10 is **identical** between Ground Truth and Transformer despite the Transformer's much lower Amount F1 — see Section 14 for why (12/14 filtered test queries produced byte-identical retrieved sets between the two sources; this is a genuine finding about the current test-query sample, not evidence that metadata quality doesn't matter). Hybrid/Mamba rows await running those checkpoints' extraction over the corpus (Section 4 — checkpoints already confirmed usable, just not yet run for this purpose).

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

**Steps 0-4 are all complete.** Step 4's core 5×2 matrix ran end-to-end (objective metrics fully trustworthy; Ragas partially trustworthy — see Section 14 for the honest breakdown).

Next: **Step 5 — Reranker ablation** (quarantined, best-config only, per the original design).

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

### 3d. Wire metadata filters into retrieval — COMPLETE

Script: `scripts/metadata_filter.py`. `MetadataRetriever` wraps the existing dense/hybrid retrievers with a metadata filter drawn from either source (`ground_truth` or `transformer`) — this is what makes the 5×2 matrix's Metadata+Vector and Metadata+Hybrid rows real, runnable systems in both metadata-source variants.

**Approach:** retrieve-then-filter. The FAISS index is flat/non-filterable at this scale (3,712 docs) — pre-filtering the vector index itself only pays off at much larger corpus sizes, so this pulls a wide candidate pool (default 100) from the base retriever, then filters down to metadata-matching chunks. Same fuzzy-matching convention as Step 3c's accuracy evaluation, so filtering behavior stays consistent with how extraction accuracy itself was measured.

**Verification exercise worth noting:** an initial test (`query="total amount"`, `filter=vendor:Walmart`) returned **zero results from both metadata sources** — investigated rather than assumed broken. Isolated the filter logic directly (confirmed 32 real Walmart chunks exist in ground truth, `matches()` correctly identifies them) before concluding the zero-result outcome was genuine, informative retrieval behavior: a generic query with no Walmart keyword simply didn't surface any of those 32 chunks within the candidate pool — not a bug. Re-tested with a query containing the actual signal (`"Walmart total amount"`) and got correct results from both sources, confirming the full pipeline works end-to-end.

**STEP 3 — COMPLETE.** All four sub-steps done: tokens added, Transformer metadata extracted (with a genuine field-coverage/error analysis), per-field accuracy computed (with a real set-based-comparison bug found and fixed), and metadata filtering wired into retrieval for both sources.

---

## 14. Step 4 progress — Core 5×2 evaluation matrix

Scripts: `scripts/build_eval_queries.py` (query set), `scripts/llm_client.py` (Qwen3/Ollama wrapper), `scripts/run_eval_matrix.py` (the harness), `scripts/compute_objective_metrics.py`, `scripts/run_ragas_eval.py`.

### 4a. Query set — COMPLETE

57 queries in `data/eval/queries.jsonl`: ~45 template-generated directly from real `chunks.jsonl` ground-truth fields (guarantees a known, verifiable target — either one `target_chunk_id` or, for metadata-targeted queries, a `target_chunk_ids` set) across question types A (factual field lookup), B (metadata-filtered lookup); ~12 hand-written for D (semantic paraphrase, no lexical overlap), E (negation/absence), C (multi-doc comparison). CORD's lack of vendor/date (by design, Section 3) is respected — those query types only sample WildReceipt/SROIE.

**Two real bugs found and fixed via spot-check before trusting the set (same discipline as Sections 3/12):**
- **Ambiguous anchor fields.** First draft anchored vendor-lookup queries on amount and date-lookup queries on vendor alone — but 5/12 amount values and every high-volume vendor collide across multiple chunks, making "the" recorded target a coin-flip among several equally-valid answers. Fixed by requiring anchors to be verified-unique (checked via a value→chunk_ids index) before use, falling back to compound anchors (vendor+amount, vendor+date) where a single field wasn't unique enough.
- **Degenerate `$0.00` "cheapest receipt" artifact.** The comparison-type query "cheapest Target receipt" initially resolved to a `$0.00` line (a void/comp-line extraction artifact, not a real purchase). Fixed by excluding non-positive amounts from the MIN selection.

A stratified `ragas_subset` flag (originally 13 queries, reduced to **10** — 2 per question type — after the Ragas cost investigation below) marks which queries get LLM generation at all; objective Recall@10/MRR need no LLM and run on the full 57.

### 4b. LLM client — COMPLETE, one major latency fix

`scripts/llm_client.py` calls Qwen3 8B via Ollama's OpenAI-compatible endpoint (the already-installed `openai` package — no new dependency). **Real finding:** Qwen3 is a "thinking" model; with thinking on, a single trivial answer took **~55s**. Ollama honors a `"think": false` field on this endpoint — passing it cut the same call to **~4s** (~13x). This was verified directly against the running Ollama instance before being trusted, not assumed from documentation.

### 4c. Eval harness — COMPLETE, one real bug found and fixed

`scripts/run_eval_matrix.py` drives all 7 system-runs (Vector-only, BM25-only, Hybrid — each once; Metadata+Vector, Metadata+Hybrid — each ×2 metadata sources) over the query set, logging every query via the existing `QueryLogger`. Writes `data/eval/run_config.json` recording the exact run configuration (systems, top_k, LLM, fusion method, pool sizes) before executing, for reproducibility.

**Real bug found during the `--limit 5` pilot run (not the full run — caught early, per this project's established practice):** `metadata_filters_applied` was being logged for `vector_only`/`bm25_only`/`hybrid` too, even though those systems never apply a filter — mislabeling unfiltered runs as filtered. Fixed to only log the filter for the two metadata-aware systems.

**Real bug found via a suspicious result (hit@10 differed between `hybrid` and `metadata_hybrid` even on queries with NO filter, where the two should be identical):** `MetadataRetriever`'s hybrid path (`metadata_filter.py`, Step 3d) fuses over a candidate pool of 100 per retriever before truncating to top-10, while the plain `hybrid` system was using the default pool of 50. Since RRF's score depends on which candidates even enter the fusion sum, a shallower pool measurably changes the top-10 even with an empty filter — a real confound, not a metadata effect. Fixed by making the plain `hybrid` row use the same `pool_size=100` for a fair comparison, and re-verified: `hybrid__none` then scored identically to `metadata_hybrid__*` on the un-filtered portion of the query set, confirming the fix.

### 4d. Objective metrics (Recall@10, MRR) — COMPLETE, fully trustworthy

`scripts/compute_objective_metrics.py` needs no LLM, so it runs on all 57 queries × 7 systems.

| System | hit@10 (single-target, n=36) | MRR |
|---|---|---|
| vector_only | 0.417 | 0.231 |
| bm25_only | 0.639 | 0.519 |
| hybrid | 0.556 | 0.385 |
| metadata_vector (GT / Transformer) | 0.417 / 0.417 | 0.231 / 0.231 |
| metadata_hybrid (GT / Transformer) | 0.556 / 0.556 | 0.385 / 0.385 |

BM25 outperforming dense/hybrid on hit@10 here reflects the query set's heavy weighting toward factual field-lookup queries with real lexical anchors (vendor names, amounts) — consistent with Step 2's finding that BM25 wins when the query's real content words are distinctive keywords, not a contradiction of it.

**Filtered-queries-only (n=14, the only queries where GT vs. Transformer metadata can possibly diverge — see Section 13d's `MetadataStore.matches()` semantics):** Recall@10 = 76.4% and Precision@10 ≈ 82% for **both** metadata sources, nearly identically. Investigated rather than accepted at face value: **12 of the 14 filtered queries produced byte-identical retrieved chunk_id lists between Ground Truth and Transformer metadata** (the other 2 differ only past the top few positions). This is a genuine finding, not a bug — these test queries were sampled from high-volume, "easy" vendors (Step 3b's extraction errors concentrate more in the AMOUNT field, which isn't used by any current filter type), so this specific slice of the eval doesn't show strong metadata-quality erosion even though Step 3c's F1 numbers show a real, meaningful gap. This is a genuine limitation of the current filtered-query sample, not evidence against RQ2 — worth a larger/more adversarial filtered-query sample in Step 6's error analysis if the linkage table's flat result needs deeper explanation.

Large-target-set queries (type E, "no vendor/date" — target = all 1,000 CORD chunks) are reported separately since Recall@10 against a 1,000-chunk target set is structurally capped at 1% for even a perfect retriever — a property of the metric at this scale, not a bug.

Full report: `data/eval/objective_metrics_report.json` (gitignored, regenerable).

### 4e. Ragas (LLM-judged) evaluation — COMPLETE with a documented, real limitation

**Real problem #1 — timeout misconfiguration, fixed.** The first full run came back with `faithfulness` and `context_precision` as `NaN` on every single system. Root cause, confirmed by measuring a single heavy judge call in isolation (~76s, CPU-bound Qwen3, full 10-chunk context): Ragas's default `RunConfig` (180s timeout, 16 concurrent workers) fires far more concurrent requests than one local Ollama instance can actually process, so most jobs queue behind each other and get killed before they even start — worst for the call-heavy metrics (ContextPrecision fires ~1 call per retrieved chunk; Faithfulness fires several per sample). Fixed with `RunConfig(timeout=600, max_workers=2)`.

**Real problem #2 — cost, addressed by scope reduction (explicit decision).** Even fixed, a full 7-system run was estimated at 2-3 hours given genuine per-call latency at this concurrency. Validated the fix cheaply first (2-sample, `max_workers=1` smoke test: 0 failures, ~6 min) before committing to the full run — the same "verify small before trusting the full run" discipline as every prior step. Rather than accept the multi-hour cost or the more invasive option (truncating ContextPrecision to fewer context chunks, which changes what's measured), reduced the `ragas_subset` from 13 to **10** queries (2 per question type) — a ~25-30% cost reduction that preserves at least minimal per-type coverage.

**Real problem #3 — a genuine, unresolved model-capability limitation, NOT fixed, explicitly documented instead of hidden.** Even after both fixes, the full reduced-subset run came back with:

| Metric | Failure rate (all 7 systems) | Trustworthy? |
|---|---|---|
| Faithfulness | 6-9 of 10 samples failed to parse (60-90%) | **No** |
| Context Recall | 3-4 of 6 samples failed to parse (50-67%) | **No** |
| Answer Relevancy | 0-5 of 10 failed (mostly 0-2) | Yes |
| Context Precision | 0-2 of 6 failed | Yes |

Root cause: `RagasOutputParserException` — Qwen3 8B (a small local model, no JSON-mode/function-calling enforcement) can't reliably produce the exact structured output Faithfulness's NLI-statement-verification and ContextRecall's classification prompts expect, even after Ragas's own `fix_output_format` retry. This is a genuine model-capability limitation, not a config bug — a decision point, not a bug to chase. **Explicit decision (confirmed):** accept AnswerRelevancy + ContextPrecision as the trustworthy Ragas signal; Faithfulness and ContextRecall are documented as unreliable with a local judge rather than reported as real numbers. This is a concrete, real-world confirmation of the original scope doc's own anticipated limitation — objective and Ragas metrics were designed as separate trust tiers specifically because LLM-judged scores carry judge-dependent risk; this is that risk materializing exactly as anticipated, not a project failure.

**Trustworthy Ragas results (AnswerRelevancy, ContextPrecision — n=10 / n=6 respectively):**

| System | Answer Relevancy | Context Precision |
|---|---|---|
| vector_only | 0.543 | 0.140 |
| bm25_only | 0.514 | 0.240 |
| hybrid | 0.561 | 0.217 |
| metadata_vector (GT) | 0.577 | 0.093 |
| metadata_vector (Transformer) | 0.599 | 0.093 |
| metadata_hybrid (GT) | 0.590 | 0.183 |
| metadata_hybrid (Transformer) | 0.681 | 0.250 |

Full report (including failure counts per metric): `data/eval/ragas_report.json` (gitignored, regenerable).

**STEP 4 — COMPLETE.** Objective metrics (Recall@10/MRR, the linkage table) are fully trustworthy and answer RQ1-3 directly. Ragas is partially trustworthy by explicit, documented decision (AnswerRelevancy/ContextPrecision usable; Faithfulness/ContextRecall are a known, real limitation of judging with a small local LLM, not silently swept under an aggregate number). Three real bugs fixed along the way (metadata_filters_applied mislabeling, hybrid pool_size confound, Ragas RunConfig timeout) plus two query-set bugs (ambiguous anchors, degenerate min-amount) — six more real bugs added to this project's running total.

---

---