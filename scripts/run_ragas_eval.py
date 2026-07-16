"""
Step 4: Ragas (LLM-judged) evaluation over the stratified subset.

Reads back the already-logged records from run_eval_matrix.py for queries
flagged `ragas_subset: true` (no re-retrieval, no re-generation needed —
the harness already produced everything this needs) and scores them with
Ragas's Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall.

Judge LLM: ChatOpenAI pointed at the same local Ollama endpoint used for
generation (scripts/llm_client.py) — one LLM integration path for the whole
project, not two. Embeddings: HuggingFaceEmbeddings("BAAI/bge-large-en-v1.5")
— the exact model already used for the FAISS index, so no extra download
and no second embedding space to reason about.

Two of the four metrics need a reference (ground_truth) answer to score
against. The semantic-paraphrase (D) and negation/absence (E) hand-written
queries deliberately have no single correct answer (see
build_eval_queries.py) — Faithfulness/AnswerRelevancy still run on them
(they only need question+answer+context), but ContextPrecision/ContextRecall
are computed on a further-restricted subset that actually has a reference.

Usage:
    python scripts/run_ragas_eval.py
"""

import argparse
import json
import os
import time

SYSTEM_RUNS = [
    ("vector_only", "none"),
    ("bm25_only", "none"),
    ("hybrid", "none"),
    ("metadata_vector", "ground_truth"),
    ("metadata_vector", "transformer"),
    ("metadata_hybrid", "ground_truth"),
    ("metadata_hybrid", "transformer"),
]

OLLAMA_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "qwen3:8b"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


def read_log(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_samples(records, require_reference):
    from ragas import SingleTurnSample

    samples = []
    for r in records:
        if r.get("generated_answer") is None:
            continue  # not in the ragas subset for this run, or generation failed
        if require_reference and not r.get("ground_truth_answer"):
            continue
        samples.append(SingleTurnSample(
            user_input=r["query_text"],
            retrieved_contexts=[c["text"] for c in r["retrieved_chunks"]],
            response=r["generated_answer"],
            reference=r.get("ground_truth_answer") or "",
        ))
    return samples


def get_judge_llm_and_embeddings():
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    chat = ChatOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", model=LLM_MODEL,
                       extra_body={"think": False})
    hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(hf_embeddings)


def get_run_config(max_workers=2):
    from ragas.run_config import RunConfig

    # Measured directly against this project's own Ollama instance: a SINGLE
    # judge call over a full 10-chunk context takes ~76s in complete
    # isolation (CPU-bound Qwen3 8B, thinking disabled). Ragas's default
    # RunConfig (timeout=180s, max_workers=16) fires 16 such calls
    # concurrently at one local, effectively single-threaded Ollama process —
    # most queue behind each other and get killed by the 180s timeout before
    # they even start, which is exactly why faithfulness (several calls per
    # sample) and context_precision (one call per context chunk, so up to 10
    # per sample) came back 100% NaN on the first run while the
    # fewer-call-per-sample metrics (answer_relevancy, context_recall) partly
    # got through. Low concurrency + a generous timeout fixes both: nothing
    # queues long enough to starve, and slow-but-real calls aren't killed.
    #
    # IMPORTANT COST NOTE (found during validation): ContextPrecision issues
    # roughly one judge call PER RETRIEVED CONTEXT CHUNK — with k=10 chunks
    # per query, that's up to 10 heavy (~76s) calls for a single sample. A
    # real full-system run at max_workers=2 can take well over an hour; test
    # with --limit on a couple of samples first before committing to a full
    # system, let alone all 7.
    return RunConfig(timeout=600, max_workers=max_workers)


def evaluate_system(records, llm, embeddings, max_workers=2, verbose=False):
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

    result = {}
    run_config = get_run_config(max_workers=max_workers)

    def col_stats(df, col):
        n_nan = int(df[col].isna().sum())
        mean = float(df[col].mean()) if n_nan < len(df) else None
        return mean, n_nan

    no_ref_samples = build_samples(records, require_reference=False)
    if no_ref_samples:
        print(f"  Faithfulness + AnswerRelevancy over {len(no_ref_samples)} samples "
              f"(no per-chunk fan-out, should be the faster half)...")
        t0 = time.monotonic()
        ds = EvaluationDataset(samples=no_ref_samples)
        scores = evaluate(dataset=ds, metrics=[Faithfulness(), AnswerRelevancy()],
                           llm=llm, embeddings=embeddings, run_config=run_config, show_progress=verbose)
        print(f"  ...done in {time.monotonic() - t0:.1f}s")
        df = scores.to_pandas()
        result["n_no_reference"] = len(no_ref_samples)
        result["faithfulness"], result["faithfulness_n_failed"] = col_stats(df, "faithfulness")
        result["answer_relevancy"], result["answer_relevancy_n_failed"] = col_stats(df, "answer_relevancy")

    ref_samples = build_samples(records, require_reference=True)
    if ref_samples:
        print(f"  ContextPrecision + ContextRecall over {len(ref_samples)} samples "
              f"(ContextPrecision fans out ~1 call per retrieved chunk — the expensive half)...")
        t0 = time.monotonic()
        ds = EvaluationDataset(samples=ref_samples)
        scores = evaluate(dataset=ds, metrics=[ContextPrecision(), ContextRecall()],
                           llm=llm, embeddings=embeddings, run_config=run_config, show_progress=verbose)
        print(f"  ...done in {time.monotonic() - t0:.1f}s")
        df = scores.to_pandas()
        result["n_with_reference"] = len(ref_samples)
        result["context_precision"], result["context_precision_n_failed"] = col_stats(df, "context_precision")
        result["context_recall"], result["context_recall_n_failed"] = col_stats(df, "context_recall")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--out", default="data/eval/ragas_report.json")
    ap.add_argument("--systems", default=None,
                     help="Comma-separated subset of system__metadata_source keys to run, "
                          "e.g. vector_only__none. Default: all 7. Existing report entries "
                          "for OTHER systems are preserved, not wiped.")
    ap.add_argument("--limit", type=int, default=None,
                     help="Cap the number of ragas-subset queries per system — use for a "
                          "quick smoke test (e.g. --limit 2) before committing to a full run.")
    ap.add_argument("--max-workers", type=int, default=2,
                     help="Ragas RunConfig concurrency. Lower (e.g. 1) for a smoke test to "
                          "get clean, unambiguous per-call timing.")
    ap.add_argument("--verbose", action="store_true",
                     help="Show ragas's own per-job progress bar (useful for a smoke test).")
    args = ap.parse_args()

    print("Setting up judge LLM (Qwen3 via Ollama, thinking disabled) and embeddings...")
    llm, embeddings = get_judge_llm_and_embeddings()

    report = {}
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            report = json.load(f)

    wanted = set(args.systems.split(",")) if args.systems else None
    for system, metadata_source in SYSTEM_RUNS:
        key = f"{system}__{metadata_source}"
        if wanted and key not in wanted:
            continue
        path = os.path.join(args.log_dir, f"{key}.jsonl")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        records = read_log(path)
        ragas_records = [r for r in records if r.get("generated_answer") is not None]
        if args.limit:
            ragas_records = ragas_records[:args.limit]
        if not ragas_records:
            print(f"{key}: no generated answers found, skipping")
            continue

        print(f"\n=== {key} ({len(ragas_records)} queries) ===")
        report[key] = evaluate_system(ragas_records, llm, embeddings,
                                       max_workers=args.max_workers, verbose=args.verbose)
        print(json.dumps(report[key], indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
