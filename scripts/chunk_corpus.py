"""
Step 1a: Chunking.

One chunk per document (receipts/invoices are short and self-contained —
no internal splitting needed). This script's real job is unifying the three
separate rag_corpus sources (wildreceipt/sroie/cord) into a single, uniformly
formatted chunks.jsonl that the rest of the pipeline (embedding, indexing,
metadata layer) builds on, with stable chunk_ids and the ground-truth BIO
tags carried through for the dual-track metadata layer (Step 3).

Usage:
    python scripts/chunk_corpus.py \
        --corpus-dir data/rag_corpus \
        --out data/chunks/chunks.jsonl
"""

import argparse
import json
import os


def load_source(path, source_name):
    records = []
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping {source_name}")
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def extract_gt_fields(tokens, tags):
    """
    Pull out the ground-truth entity spans (vendor/date/amount) as clean
    strings, for use as structured metadata in Step 3 — not just raw
    BIO tags, but the actual reconstructed field values.
    """
    fields = {"NAME": [], "DATE": [], "AMOUNT": []}
    current_tag, current_words = None, []

    def flush():
        if current_tag and current_words:
            fields[current_tag].append(" ".join(current_words))

    for tok, tag in zip(tokens, tags):
        if tag == "O":
            flush()
            current_tag, current_words = None, []
        elif tag.startswith("B-"):
            flush()
            current_tag = tag[2:]
            current_words = [tok]
        elif tag.startswith("I-") and tag[2:] == current_tag:
            current_words.append(tok)
        else:
            # I- tag with no matching B- (shouldn't happen, but don't crash)
            flush()
            current_tag, current_words = None, []
    flush()

    return {
        "vendor": fields["NAME"][0] if fields["NAME"] else None,
        "date": fields["DATE"][0] if fields["DATE"] else None,
        "amount": fields["AMOUNT"][0] if fields["AMOUNT"] else None,
        # keep all matches too, in case a doc has multiple amount spans
        # (e.g. subtotal + tax + total all tagged AMOUNT)
        "all_vendors": fields["NAME"],
        "all_dates": fields["DATE"],
        "all_amounts": fields["AMOUNT"],
    }


def build_chunk(record, source, chunk_idx):
    tokens = record.get("tokens", [])
    tags = record.get("tags", [])
    text = record.get("text", "")
    gt = extract_gt_fields(tokens, tags)

    return {
        "chunk_id": f"chunk_{chunk_idx:06d}",
        "doc_id": record.get("doc_id"),
        "source": source,
        "text": text,
        "tokens": tokens,  # kept for Step 3b: feeding the Transformer checkpoint
        "char_len": len(text),
        "word_len": len(tokens),
        "ground_truth": gt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="data/rag_corpus")
    ap.add_argument("--out", default="data/chunks/chunks.jsonl")
    args = ap.parse_args()

    sources = ["wildreceipt", "sroie", "cord"]
    all_chunks = []

    for source in sources:
        path = os.path.join(args.corpus_dir, f"{source}.jsonl")
        print(f"Loading {source} from {path}...")
        records = load_source(path, source)
        print(f"  {len(records)} documents")
        for record in records:
            chunk = build_chunk(record, source, len(all_chunks))
            all_chunks.append(chunk)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_chunks)} chunks -> {args.out}")

    if not all_chunks:
        print("\nERROR: No chunks were produced — check that --corpus-dir "
              "points at the directory containing wildreceipt.jsonl, "
              "sroie.jsonl, and cord.jsonl (data/rag_corpus/ in the "
              "document-intelligence-rag repo, not the old extraction repo).")
        return

    # quick sanity stats
    with_vendor = sum(1 for c in all_chunks if c["ground_truth"]["vendor"])
    with_date = sum(1 for c in all_chunks if c["ground_truth"]["date"])
    with_amount = sum(1 for c in all_chunks if c["ground_truth"]["amount"])
    char_lens = [c["char_len"] for c in all_chunks]
    print(f"\nGround-truth field coverage:")
    print(f"  vendor: {with_vendor}/{len(all_chunks)} ({with_vendor/len(all_chunks)*100:.1f}%)")
    print(f"  date:   {with_date}/{len(all_chunks)} ({with_date/len(all_chunks)*100:.1f}%)")
    print(f"  amount: {with_amount}/{len(all_chunks)} ({with_amount/len(all_chunks)*100:.1f}%)")
    print(f"\nChunk length (chars): min={min(char_lens)}, max={max(char_lens)}, "
          f"avg={sum(char_lens)/len(char_lens):.0f}")

    # flag any suspiciously empty or tiny chunks — worth a look before embedding
    tiny = [c for c in all_chunks if c["char_len"] < 20]
    if tiny:
        print(f"\nWARNING: {len(tiny)} chunks under 20 chars — sample:")
        for c in tiny[:5]:
            print(f"  {c['chunk_id']} ({c['source']}, {c['doc_id']}): {c['text']!r}")


if __name__ == "__main__":
    main()
