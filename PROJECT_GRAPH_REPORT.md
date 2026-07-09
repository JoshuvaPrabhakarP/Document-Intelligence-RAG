# Enterprise Document Intelligence RAG — GRAPH REPORT
> Living status document. Update this whenever a decision is made, a gate is passed, or scope changes — treat it the way `GRAPH_REPORT.md` was treated in the extraction project: the canonical source of truth, not the chat history.

**This is a fresh, standalone project with its own new GitHub repo — not a continuation phase of `hybrid-doc-extractor`.** It imports specific reusable assets (dataset, trained Transformer checkpoint, inference-loading code) from that prior project, but has its own independent phase numbering starting at Phase 1.

**Prior project (asset source only):** `github.com/JoshuvaPrabhakarP/hybrid-doc-extractor` (its Phases 1-8, complete)
**This project:** repo created and live — `github.com/JoshuvaPrabhakarP/Document-Intelligence-RAG`
**Current stage:** This project's Phase 1, Step 0 — Environment & Data Prep (install list complete, data reconstruction pending)

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
| 0 | Environment & data prep | 🔶 In progress (this document) |
| 1 | Vector-only baseline + structured logging | ⏳ Blocked on Step 0 |
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

**Status: COMPLETE.** `data/rag_corpus/{wildreceipt,sroie}.jsonl` written — 1,739 WildReceipt + 973 SROIE = 2,712 documents (CORD not yet included — see open item below).

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
**CORD not included.** Original dataset composition was WildReceipt (1,739) + SROIE (973) + CORD (972) = 3,684. Only the first two are reconstructed so far. CORD's `ground_truth` field is itemized JSON, not box coordinates — needs a different loader, not a plug-in to the existing box-based functions. Location of local CORD raw data (if any) not yet confirmed.

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
| Local LLM choice | Qwen 3 vs Llama 3.x — **not yet decided** |
| VRAM budget | Extractor (old venv) + BGE-large + local LLM (new venv) concurrently is tight on 8GB. **Not yet decided:** quantize-and-swap on GPU vs. LLM-on-CPU. Current lean: LLM on CPU, since generation only runs during the eval/answer step. |

**Open risk not yet closed:** `import mamba_ssm` smoke test in the *old* venv on the current gcc-13 toolchain — determines whether Mamba/Hybrid checkpoints are usable now or deferred until the linkage table step.

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

An untracked `app/` directory exists in the *extraction* repo (`app/inference.py`, `app/dashboard.py`, `app/charts.py` — distinct from the git-tracked `app_gradio/`), plus loose `evaluation/`, `get_real_examples.py`, `peek_test_sample.py`, and two zips (`checkpoints_phase5.zip`, `evaluation_phase6.zip`). None committed. **Not yet triaged** — unclear if WIP scaffold worth carrying into the new repo or dead leftovers. Should be resolved before copying anything into `document-intelligence-rag`.

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

---

## 9. Gating checklist before Step 1 can start

- [x] New venv created (`~/document-intelligence-rag/venv`, Python 3.12)
- [x] Step 0 stack installed in new venv (Section 6) — two real fixes applied: `ragas`/`langchain-community` shim, `torch` CUDA build pin
- [x] New repo created on GitHub and pushed (`Document-Intelligence-RAG`)
- [x] Reading-order reconstruction script written + verified (8-doc spot-check + 2 corpus-wide scans, 3 real bugs found and fixed)
- [ ] BIO tags re-associated with reconstructed text — **done as part of reconstruction itself** (tags are generated directly from raw sources, not re-associated after the fact — see Section 3)
- [ ] CORD integration (not yet started — open item, Section 3)
- [ ] `import mamba_ssm` smoke test in **old venv** on gcc-13 toolchain — determines Mamba/Hybrid checkpoint availability now vs. deferred
- [ ] LLM placement decision made (CPU vs. GPU-swap) — doesn't need to be final, needs to be explicit
- [ ] `app/` vs `app_gradio/` vs loose `evaluation/`/zips triaged in old repo — keep, discard, or archive

---

## 10. Next immediate action (as of this update)

Reading-order reconstruction script is the highest-priority unblock — nothing downstream can proceed without it. `app/` triage is a quick, low-risk parallel task.

---

*Update this file after every resolved decision or completed gate — don't rely on chat history to reconstruct project state in future sessions.*