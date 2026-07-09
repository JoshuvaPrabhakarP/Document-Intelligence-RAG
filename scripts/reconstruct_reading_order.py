"""
Reading-order reconstruction script.

Rebuilds human-readable, reading-order-correct document text from the RAW
SROIE and WildReceipt sources (which retain per-box polygon coordinates),
instead of the flattened data/processed/*.jsonl files (which lost both
whitespace and box order during the original BIO conversion).

Usage (run from repo root, either venv — stdlib only, no torch/faiss needed):
    python scripts/reconstruct_reading_order.py \
        --wildreceipt-dir /path/to/data/raw/wildreceipt/wildreceipt \
        --sroie-dir /path/to/data/raw/sroie/SROIE2019 \
        --out-dir data/rag_corpus \
        --verify-sample 5

Output: data/rag_corpus/wildreceipt.jsonl and data/rag_corpus/sroie.jsonl
Each line: {"doc_id": str, "source": str, "text": str,
            "tokens": [str, ...], "tags": [str, ...]}
"""

import argparse
import difflib
import glob
import json
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def box_top_left_y(coords):
    """coords = [x1,y1,x2,y2,x3,y3,x4,y4] (4-point polygon). Return min y."""
    ys = coords[1::2]
    return min(ys)


def box_left_x(coords):
    xs = coords[0::2]
    return min(xs)


def box_height(coords):
    ys = coords[1::2]
    return max(ys) - min(ys)


def sort_boxes_reading_order(boxes):
    """
    boxes: list of dicts with a 'coords' key ([x1,y1,...,x4,y4]).
    Clusters boxes into lines by y-proximity, sorts each line left-to-right,
    then returns (ordered_boxes, line_break_after_indices) so callers can
    join with '\n' between lines and ' ' within a line.
    """
    if not boxes:
        return [], []

    # Sort by top y first to establish rough vertical order
    boxes = sorted(boxes, key=lambda b: box_top_left_y(b["coords"]))

    avg_height = sum(box_height(b["coords"]) for b in boxes) / len(boxes)
    y_tolerance = max(avg_height * 0.6, 5)

    lines = []  # list of list-of-boxes
    current_line = [boxes[0]]
    current_y = box_top_left_y(boxes[0]["coords"])

    for b in boxes[1:]:
        y = box_top_left_y(b["coords"])
        if abs(y - current_y) <= y_tolerance:
            current_line.append(b)
        else:
            lines.append(current_line)
            current_line = [b]
            current_y = y
    lines.append(current_line)

    # sort each line left-to-right
    for line in lines:
        line.sort(key=lambda b: box_left_x(b["coords"]))

    ordered = []
    line_break_after = set()
    for line in lines:
        ordered.extend(line)
        line_break_after.add(len(ordered) - 1)

    return ordered, line_break_after


def boxes_to_text_tokens_tags(ordered_boxes, line_break_after):
    """
    Build the reconstructed text string plus word-level tokens/tags.
    Each box's text is whitespace-tokenized; first word gets B-<TAG>,
    subsequent words in the same box get I-<TAG> (unless tag is 'O').
    """
    text_parts = []
    tokens = []
    tags = []

    for i, b in enumerate(ordered_boxes):
        box_text = b["text"].strip()
        if not box_text:
            continue
        words = box_text.split()

        if "word_tags" in b:
            # SROIE path: precise per-word tags from span matching
            box_tags = b["word_tags"]
            # defensive: pad/truncate in case of any tokenization mismatch
            if len(box_tags) != len(words):
                box_tags = (box_tags + ["O"] * len(words))[:len(words)]
        else:
            # WildReceipt path: whole box is one field, B- first word / I- rest
            label = b.get("tag", "O")
            box_tags = []
            for w_idx in range(len(words)):
                if label == "O":
                    box_tags.append("O")
                else:
                    box_tags.append(("B-" if w_idx == 0 else "I-") + label)

        for w, t in zip(words, box_tags):
            tokens.append(w)
            tags.append(t)

        text_parts.append(box_text)
        if i in line_break_after:
            text_parts.append("\n")
        else:
            text_parts.append(" ")

    text = "".join(text_parts).strip()
    return text, tokens, tags


# ---------------------------------------------------------------------------
# WildReceipt
# ---------------------------------------------------------------------------

WILDRECEIPT_LABEL_MAP = {
    1: "NAME",    # Store_name_value
    7: "DATE",    # Date_value
    17: "AMOUNT", # Subtotal_value
    19: "AMOUNT", # Tax_value
    23: "AMOUNT", # Total_value
}


def load_wildreceipt(wildreceipt_dir):
    docs = []
    for split_file in ["train.txt", "test.txt"]:
        path = os.path.join(wildreceipt_dir, split_file)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = f"wildreceipt_{split_file}_{line_no}"
                boxes = []
                for ann in record.get("annotations", []):
                    coords = ann.get("box")
                    txt = ann.get("text", "")
                    label_id = ann.get("label")
                    if not coords or not txt:
                        continue
                    boxes.append({
                        "coords": coords,
                        "text": txt,
                        "tag": WILDRECEIPT_LABEL_MAP.get(label_id, "O"),
                    })
                if boxes:
                    docs.append({"doc_id": doc_id, "boxes": boxes})
    return docs


# ---------------------------------------------------------------------------
# SROIE
# ---------------------------------------------------------------------------

FIELD_TO_TAG = {"company": "NAME", "date": "DATE", "total": "AMOUNT"}
FUZZY_THRESHOLD = 0.6


def parse_sroie_box_file(path):
    """SROIE box files: x1,y1,x2,y2,x3,y3,x4,y4,transcript per line."""
    boxes = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 8)
            if len(parts) < 9:
                continue
            try:
                coords = [float(p) for p in parts[:8]]
            except ValueError:
                continue
            text = parts[8]
            boxes.append({"coords": coords, "text": text, "tag": "O"})
    return boxes


def fuzzy_match_score(a, b):
    a_norm, b_norm = a.lower().strip(), b.lower().strip()
    if not a_norm or not b_norm:
        return 0.0
    shorter, longer = sorted([a_norm, b_norm], key=len)
    # Guard: the substring shortcut is only meaningful when the shorter
    # string carries enough information to be a real match, not a stray
    # single digit/character that trivially appears inside a longer date
    # or amount string (e.g. "1" inside "12-01-19").
    if shorter in longer and len(shorter) >= 4:
        return 1.0
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


def normalize_nospace(s):
    """Strip all whitespace for comparison — neutralizes OCR tokenization
    differences like '.K(TAMAN' (box) vs '.K (TAMAN' (entities.txt)."""
    return re.sub(r"\s+", "", s.lower())


def find_best_company_span(value, boxes):
    """
    Company names can (a) be tokenized differently between entities.txt and
    the OCR box text, and (b) span multiple consecutive boxes/lines. Neither
    is solvable with a fixed-word-count single-box window, so this compares
    whitespace-normalized text across sliding windows of 1-3 consecutive
    boxes instead. Returns (start_box_idx, end_box_idx_inclusive, score).

    Tie-breaking: prefers the SHORTEST window on equal/near-equal scores.
    Without this, a longer window that happens to swallow an unrelated
    adjacent box (e.g. a recurring signature/watermark line concatenated
    with no separator directly in front of the real company box) can score
    a coincidental perfect match via the substring shortcut, and — since
    windows are scanned in ascending start order — that bloated match gets
    found before the correct, tighter one and is never displaced by a
    later tie.
    """
    value_norm = normalize_nospace(value)
    if not value_norm:
        return None, None, 0.0

    best_start, best_end, best_score = None, None, -1.0
    best_len = None
    EPS = 1e-9

    for start in range(len(boxes)):
        concatenated = ""
        for end in range(start, min(start + 3, len(boxes))):
            btext = boxes[end]["text"]
            if len(btext.strip()) < 1:
                continue
            concatenated += btext
            concat_norm = normalize_nospace(concatenated)
            if not concat_norm:
                continue

            score = difflib.SequenceMatcher(None, value_norm, concat_norm).ratio()
            shorter, longer = sorted([value_norm, concat_norm], key=len)
            if len(shorter) >= 4 and shorter in longer:
                score = max(score, 1.0)

            window_len = end - start + 1
            better = (
                score > best_score + EPS
                or (abs(score - best_score) <= EPS and (best_len is None or window_len < best_len))
            )
            if better:
                best_score, best_start, best_end, best_len = score, start, end, window_len

    return best_start, best_end, best_score


def find_best_span(value, box_words):
    """
    Slide a window the same word-length as `value` across box_words,
    return (start_idx, span_len, score) for the best-matching span —
    so we tag only the words that actually correspond to the entity
    value, not the entire box (which may contain trailing unrelated
    text like register/terminal codes on the same OCR line).
    """
    value_words = value.split()
    span_len = max(len(value_words), 1)
    best_start, best_score = None, 0.0
    for start in range(len(box_words) - span_len + 1):
        candidate = " ".join(box_words[start:start + span_len])
        score = fuzzy_match_score(value, candidate)
        if score > best_score:
            best_score, best_start = score, start
    return best_start, span_len, best_score


def tag_sroie_boxes(boxes, entities):
    match_log = {}
    for b in boxes:
        b["word_tags"] = ["O"] * len(b["text"].split())

    # --- company: multi-box, whitespace-normalized matching ---
    value = entities.get("company")
    if value:
        start_box, end_box, score = find_best_company_span(value, boxes)
        if start_box is not None and score >= FUZZY_THRESHOLD:
            first_word_tagged = False
            for bi in range(start_box, end_box + 1):
                wt = boxes[bi]["word_tags"]
                for wi in range(len(wt)):
                    wt[wi] = ("B-NAME" if not first_word_tagged else "I-NAME")
                    first_word_tagged = True
            match_log["company"] = f"matched ({score:.2f}, boxes {start_box}-{end_box})"
        else:
            match_log["company"] = f"no_match (best={score:.2f})"
    else:
        match_log["company"] = "no_value_in_entities"

    # --- date / total: single-box, exact-word-count span matching ---
    for field, tag in [("date", "DATE"), ("total", "AMOUNT")]:
        value = entities.get(field)
        if not value:
            match_log[field] = "no_value_in_entities"
            continue

        best_box_idx, best_start, best_span, best_score = None, None, None, 0.0
        for i, b in enumerate(boxes):
            words = b["text"].split()
            if len(words) < 1 or len(b["text"].strip()) < 3:
                continue
            start, span, score = find_best_span(value, words)
            if start is not None and score > best_score:
                best_box_idx, best_start, best_span, best_score = i, start, span, score

        if best_box_idx is not None and best_score >= FUZZY_THRESHOLD:
            wt = boxes[best_box_idx]["word_tags"]
            for offset in range(best_span):
                idx = best_start + offset
                if idx < len(wt):
                    wt[idx] = ("B-" if offset == 0 else "I-") + tag
            match_log[field] = f"matched ({best_score:.2f})"
        else:
            match_log[field] = f"no_match (best={best_score:.2f})"

    return match_log


def load_sroie(sroie_dir):
    docs = []
    match_stats = {"company": [0, 0], "date": [0, 0], "total": [0, 0]}

    for split in ["train", "test"]:
        box_dir = os.path.join(sroie_dir, split, "box")
        entities_dir = os.path.join(sroie_dir, split, "entities")
        if not os.path.isdir(box_dir) or not os.path.isdir(entities_dir):
            continue

        for box_path in sorted(glob.glob(os.path.join(box_dir, "*.txt"))):
            stem = Path(box_path).stem
            entities_path = os.path.join(entities_dir, stem + ".txt")
            if not os.path.exists(entities_path):
                continue

            boxes = parse_sroie_box_file(box_path)
            if not boxes:
                continue

            try:
                with open(entities_path, "r", encoding="utf-8", errors="ignore") as f:
                    entities = json.load(f)
            except json.JSONDecodeError:
                continue

            match_log = tag_sroie_boxes(boxes, entities)
            for field in match_stats:
                match_stats[field][1] += 1
                if match_log.get(field, "").startswith("matched"):
                    match_stats[field][0] += 1

            docs.append({"doc_id": f"sroie_{split}_{stem}", "boxes": boxes})

    print("SROIE fuzzy-match rates:")
    for field, (matched, total) in match_stats.items():
        pct = (matched / total * 100) if total else 0.0
        print(f"  {field}: {matched}/{total} ({pct:.1f}%)")

    return docs


# ---------------------------------------------------------------------------
# CORD (naver-clova-ix/cord-v2)
# ---------------------------------------------------------------------------
# Unlike SROIE/WildReceipt, cord-v2 has NO per-box coordinates — it's the
# Donut-style simplified format: image + a structured ground_truth JSON.
# There's nothing to reconstruct reading order FROM, so this path is
# deliberately simpler: build readable text directly from the structured
# fields (already in natural line order), and tag only the total amount —
# no fuzzy matching needed since we generate the text ourselves and know
# exactly where the value lands. CORD has no native date/company field
# (confirmed in the original project's Phase 2 log), so NAME/DATE stay "O"
# for every CORD document — matching the field's real absence, not faking
# labels that don't exist in the source.

def extract_cord_fields(gt_parse):
    menu = gt_parse.get("menu", [])
    if isinstance(menu, dict):
        menu = [menu]
    elif not isinstance(menu, list):
        menu = []

    total = gt_parse.get("total", {})
    if isinstance(total, list):
        total = total[0] if total else {}
    if not isinstance(total, dict):
        total = {}
    total_price = total.get("total_price")

    return menu, total_price


def build_cord_record(doc_id, ground_truth_str):
    try:
        gt = json.loads(ground_truth_str)
    except (json.JSONDecodeError, TypeError):
        return None
    gt_parse = gt.get("gt_parse", gt)
    menu, total_price = extract_cord_fields(gt_parse)

    tokens, tags, text_lines = [], [], []

    for item in menu:
        if not isinstance(item, dict):
            continue
        parts = []
        for key in ("nm", "cnt", "price"):
            val = item.get(key)
            if val:
                parts.append(str(val).strip())
        if not parts:
            continue
        line = " ".join(parts)
        text_lines.append(line)
        for w in line.split():
            tokens.append(w)
            tags.append("O")

    if total_price:
        total_str = str(total_price).strip()
        line = f"Total: {total_str}"
        text_lines.append(line)
        tokens.append("Total:")
        tags.append("O")
        for w_idx, w in enumerate(total_str.split()):
            tokens.append(w)
            tags.append("B-AMOUNT" if w_idx == 0 else "I-AMOUNT")

    if not tokens:
        return None

    return {
        "doc_id": doc_id,
        "source": "cord",
        "text": "\n".join(text_lines),
        "tokens": tokens,
        "tags": tags,
    }


def load_cord(dataset_name="naver-clova-ix/cord-v2"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # already cached, no network needed
    try:
        from datasets import load_dataset
    except ImportError:
        print("  'datasets' package not installed in this venv — skipping CORD.")
        return []

    records = []
    try:
        ds_dict = load_dataset(dataset_name)
    except Exception as e:
        print(f"  Could not load {dataset_name}: {e}")
        return []

    for split_name, split_ds in ds_dict.items():
        for i, example in enumerate(split_ds):
            gt_str = example.get("ground_truth")
            if not gt_str:
                continue
            doc_id = f"cord_{split_name}_{i}"
            record = build_cord_record(doc_id, gt_str)
            if record:
                records.append(record)

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_records(docs, source_name):
    records = []
    for doc in docs:
        ordered, line_breaks = sort_boxes_reading_order(doc["boxes"])
        text, tokens, tags = boxes_to_text_tokens_tags(ordered, line_breaks)
        if not tokens:
            continue
        records.append({
            "doc_id": doc["doc_id"],
            "source": source_name,
            "text": text,
            "tokens": tokens,
            "tags": tags,
        })
    return records


def write_jsonl(records, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_verification_sample(records, n):
    print(f"\n--- Verification sample ({min(n, len(records))} of {len(records)} docs) ---")
    for r in records[:n]:
        print(f"\n[{r['doc_id']}]")
        print(r["text"])
        tagged = [(t, g) for t, g in zip(r["tokens"], r["tags"]) if g != "O"]
        print(f"  Non-O tags: {tagged}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wildreceipt-dir", required=True)
    ap.add_argument("--sroie-dir", required=True)
    ap.add_argument("--out-dir", default="data/rag_corpus")
    ap.add_argument("--verify-sample", type=int, default=5)
    ap.add_argument("--skip-cord", action="store_true")
    ap.add_argument("--cord-dataset", default="naver-clova-ix/cord-v2")
    args = ap.parse_args()

    print("Loading WildReceipt...")
    wr_docs = load_wildreceipt(args.wildreceipt_dir)
    print(f"  {len(wr_docs)} documents")
    wr_records = build_records(wr_docs, "wildreceipt")
    write_jsonl(wr_records, os.path.join(args.out_dir, "wildreceipt.jsonl"))
    print(f"  Wrote {len(wr_records)} records -> {args.out_dir}/wildreceipt.jsonl")

    print("\nLoading SROIE...")
    sroie_docs = load_sroie(args.sroie_dir)
    print(f"  {len(sroie_docs)} documents")
    sroie_records = build_records(sroie_docs, "sroie")
    write_jsonl(sroie_records, os.path.join(args.out_dir, "sroie.jsonl"))
    print(f"  Wrote {len(sroie_records)} records -> {args.out_dir}/sroie.jsonl")

    cord_records = []
    if not args.skip_cord:
        print("\nLoading CORD...")
        cord_records = load_cord(args.cord_dataset)
        print(f"  {len(cord_records)} documents")
        write_jsonl(cord_records, os.path.join(args.out_dir, "cord.jsonl"))
        print(f"  Wrote {len(cord_records)} records -> {args.out_dir}/cord.jsonl")

    total = len(wr_records) + len(sroie_records) + len(cord_records)
    print(f"\nTotal documents across all sources: {total}")

    if args.verify_sample > 0:
        print_verification_sample(wr_records, args.verify_sample)
        print_verification_sample(sroie_records, args.verify_sample)
        if cord_records:
            print_verification_sample(cord_records, args.verify_sample)


if __name__ == "__main__":
    main()