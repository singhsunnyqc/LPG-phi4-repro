#!/usr/bin/env python3
"""Gate-only sanity-slice preparer for the LPG 40k training data.

WHAT THIS DOES
  1. Validates the SOURCE jsonl (parity, no-empty, no-silent-mislabel, no-dups).
  2. Reports the raw label balance and the >max_token_num drop rate on the FULL set.
  3. Writes a seeded N-record slice for sanity runs.

WHAT THIS DOES *NOT* DO
  It NEVER modifies, relabels, drops, rebalances, or curates the source. The slice is
  a verbatim random subset (seeded shuffle, first N). The output is NOT token-filtered:
  the training data loader (train.py) applies the >max_token_num filter itself, so
  filtering here would double-apply it and silently shrink the slice. --max_token_num
  is REPORTING ONLY.

LOGIC REUSE
  The label and verdict-parse logic is the repo's own. We first try to import the real
  functions (train.py guards training behind `if __name__ == "__main__"`, so importing
  it is side-effect free). If that import fails (e.g. torch/peft not present in the
  current env), we fall back to extracting the *same* functions verbatim from source via
  AST and exec'ing them unchanged -- the logic is never hand-rewritten.

Read-only on the source; writes only --out_path. No network beyond the tokenizer load.
"""

import argparse
import ast
import hashlib
import json
import os
import random
import sys
from typing import Dict, List, Optional

TRAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../training
TRAIN_PY = os.path.join(TRAIN_DIR, "train.py")
MODEL_PY = os.path.join(TRAIN_DIR, "src", "model.py")


# --------------------------------------------------------------------------------------
# Repo logic: import the real functions, else AST-extract them verbatim (never rewrite).
# --------------------------------------------------------------------------------------
def load_repo_functions():
    if TRAIN_DIR not in sys.path:
        sys.path.insert(0, TRAIN_DIR)
    try:
        from train import _build_answer_text  # noqa: E402
        from src.model import (  # noqa: E402
            extract_json_verdict,
            extract_simple_verdict,
            extract_tagged_block,
        )
        return (
            "import",
            _build_answer_text,
            extract_json_verdict,
            extract_simple_verdict,
            extract_tagged_block,
        )
    except Exception as exc:  # heavy deps missing -> AST fallback
        sys.stderr.write(f"[info] direct import failed ({exc!r}); using AST fallback.\n")

    def grab(path, names):
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
        out = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                out[node.name] = ast.get_source_segment(src, node)
        missing = names - set(out)
        if missing:
            raise RuntimeError(f"AST fallback could not find {missing} in {path}")
        return out

    ns: Dict[str, object] = {"re": __import__("re"), "json": json,
                             "List": List, "Optional": Optional, "Dict": Dict}
    # _tag_variants is a private dependency of extract_tagged_block.
    for body in grab(MODEL_PY, {
        "_tag_variants", "extract_tagged_block",
        "extract_json_verdict", "extract_simple_verdict",
    }).values():
        exec(body, ns)
    for body in grab(TRAIN_PY, {"_build_answer_text"}).values():
        exec(body, ns)
    return (
        "ast",
        ns["_build_answer_text"],
        ns["extract_json_verdict"],
        ns["extract_simple_verdict"],
        ns["extract_tagged_block"],
    )


def confirm_filter_string_from_code():
    """Read the 800-filter expression out of train.py and verify it still matches our
    assumption. We do NOT hardcode the filter; if upstream changes it, this fails loud."""
    src = open(TRAIN_PY, encoding="utf-8").read()
    needles = [
        'annotation_input + "\\n" + generated_reasoning',
        "add_special_tokens=False",
        "> training_args.max_token_num",
    ]
    for n in needles:
        if n not in src:
            raise RuntimeError(
                "FAIL: filter expression in train.py no longer matches the expected form "
                f"(missing snippet: {n!r}). Refusing to guess -- update this script to match."
            )
    print("[ok] confirmed 800-filter from train.py: "
          'len(tok.encode(annotation_input + "\\n" + generated_reasoning, '
          "add_special_tokens=False)) > max_token_num  -> dropped")


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def verdict_resolves(gen, extract_json_verdict, extract_simple_verdict, extract_tagged_block):
    """Mirror of _build_answer_text's verdict resolution (train.py:169-176). Returns True
    if any parser yields a verdict; False means the silent-safe fallback would fire."""
    verdict = extract_json_verdict(gen)
    if verdict is None:
        body = extract_tagged_block(gen, "Output")
        if body:
            verdict = extract_json_verdict(body)
            if verdict is None:
                verdict = extract_simple_verdict(body)
    return verdict is not None


def label_bucket(label):
    if label == "safe":
        return "safe"
    if label == "unsafe":
        return "bare_unsafe"        # unsafe with NO policy index
    if label.startswith("unsafe, policy"):
        return "unsafe_with_idx"
    return "other"


def main():
    ap = argparse.ArgumentParser(description="Gate-only LPG sanity-slice preparer.")
    ap.add_argument("--data_path", required=True, help="source jsonl")
    ap.add_argument("--out_path", required=True, help="where to write the slice")
    ap.add_argument("--n", required=True, type=int, help="records in the slice")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--model_name", default="microsoft/Phi-4-mini-instruct")
    ap.add_argument("--max_token_num", type=int, default=800,
                    help="REPORTING ONLY -- the output is NOT filtered by it.")
    ap.add_argument("--expect_records", type=int, default=40041)
    args = ap.parse_args()

    print(f"[info] data_path     = {args.data_path}")
    print(f"[info] out_path      = {args.out_path}")
    print(f"[info] n / seed      = {args.n} / {args.seed}")
    print(f"[info] model_name    = {args.model_name}")
    print(f"[info] max_token_num = {args.max_token_num} (reporting only)")

    source, _build_answer_text, extract_json_verdict, extract_simple_verdict, extract_tagged_block = \
        load_repo_functions()
    print(f"[ok] repo logic loaded via: {source}")
    confirm_filter_string_from_code()

    # ---- load + parity guard ---------------------------------------------------------
    records = read_jsonl(args.data_path)
    if len(records) != args.expect_records:
        print(f"FAIL: record-count parity -- got {len(records)}, expected {args.expect_records}. "
              "Source file does not match the EDA baseline.")
        sys.exit(1)
    print(f"[ok] parity: {len(records)} records == expected {args.expect_records}")

    # ---- HARD validation (fail loud) -------------------------------------------------
    empty_idx, allfail_idx = [], []
    seen, dup_idx = {}, []
    for i, r in enumerate(records):
        ai = r.get("annotation_input", "") or ""
        gen = r.get("generated_reasoning", "") or ""
        if not ai or not gen:
            empty_idx.append(i)
        if not verdict_resolves(gen, extract_json_verdict, extract_simple_verdict, extract_tagged_block):
            allfail_idx.append(i)
        h = hashlib.md5(json.dumps(r, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        if h in seen:
            dup_idx.append(i)
        else:
            seen[h] = i

    failures = []
    if empty_idx:
        failures.append(f"{len(empty_idx)} record(s) with empty annotation_input/generated_reasoning "
                        f"(e.g. indices {empty_idx[:10]})")
    if allfail_idx:
        failures.append(f"{len(allfail_idx)} record(s) where ALL verdict parsers fail -> silent-safe "
                        f"mislabel (e.g. indices {allfail_idx[:10]})")
    if dup_idx:
        failures.append(f"{len(dup_idx)} exact-duplicate record(s) (e.g. indices {dup_idx[:10]})")
    if failures:
        print("FAIL: source validation failed:")
        for f in failures:
            print("   - " + f)
        sys.exit(1)
    print("[ok] validation: 0 empty, 0 all-parsers-failed, 0 exact duplicates")

    # ---- soft warnings ----------------------------------------------------------------
    bare_unsafe_idx, ts_missing_idx = [], []
    for i, r in enumerate(records):
        gen = r.get("generated_reasoning", "") or ""
        if label_bucket(_build_answer_text(r, gen)) == "bare_unsafe":
            bare_unsafe_idx.append(i)
        ts = r.get("teacher_summaries") or {}
        if ts.get("intent_summary") is None or ts.get("risk_summary") is None:
            ts_missing_idx.append(i)
    if bare_unsafe_idx:
        print(f"[warn] {len(bare_unsafe_idx)} unsafe record(s) lack a policy index "
              f"(e.g. {bare_unsafe_idx[:10]})")
    if ts_missing_idx:
        print(f"[warn] {len(ts_missing_idx)} record(s) missing intent_summary/risk_summary "
              f"(e.g. {ts_missing_idx[:10]})")
    if not bare_unsafe_idx and not ts_missing_idx:
        print("[ok] no warnings (all unsafe have indices; teacher_summaries complete)")

    # ---- tokenizer (matches train.py: use_fast=False) --------------------------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    print(f"[ok] tokenizer: {type(tok).__name__} (use_fast=False)")

    def tok_len(r):
        ai = r.get("annotation_input", "") or ""
        gen = r.get("generated_reasoning", "") or ""
        return len(tok.encode(ai + "\n" + gen, add_special_tokens=False))

    def is_safe(r):
        return _build_answer_text(r, r.get("generated_reasoning", "") or "") == "safe"

    # ---- FULL-set report -------------------------------------------------------------
    n_total = len(records)
    full_safe = sum(1 for r in records if is_safe(r))
    full_unsafe = n_total - full_safe
    dropped = sum(1 for r in records if tok_len(r) > args.max_token_num)
    survive = n_total - dropped
    print("\n===== FULL SET =====")
    print(f"total                 : {n_total}")
    print(f"raw safe / unsafe     : {full_safe} ({100*full_safe/n_total:.2f}%) / "
          f"{full_unsafe} ({100*full_unsafe/n_total:.2f}%)")
    print(f">{args.max_token_num} dropped         : {dropped} ({100*dropped/n_total:.2f}%)")
    print(f"surviving (<= {args.max_token_num})    : {survive} ({100*survive/n_total:.2f}%)")

    # ---- seeded slice (verbatim subset; NOT token-filtered) --------------------------
    if args.n > n_total:
        print(f"[warn] requested n={args.n} > available {n_total}; using all {n_total}.")
    take = min(args.n, n_total)
    shuffled = list(records)
    random.seed(args.seed)
    random.shuffle(shuffled)
    sliced = shuffled[:take]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or ".", exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as fh:
        for r in sliced:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[ok] wrote {len(sliced)} records -> {args.out_path}")

    # ---- slice report (raw + realized post-800) --------------------------------------
    s_safe = sum(1 for r in sliced if is_safe(r))
    s_unsafe = len(sliced) - s_safe
    valid = [r for r in sliced if tok_len(r) <= args.max_token_num]
    v_safe = sum(1 for r in valid if is_safe(r))
    v_unsafe = len(valid) - v_safe
    n_valid = len(valid)
    print("\n===== SLICE =====")
    print(f"slice size            : {len(sliced)}")
    print(f"raw safe / unsafe     : {s_safe} ({100*s_safe/len(sliced):.2f}%) / "
          f"{s_unsafe} ({100*s_unsafe/len(sliced):.2f}%)")
    print(f"valid after >{args.max_token_num}    : {n_valid} "
          f"({100*n_valid/len(sliced):.2f}% of slice)  <- the real training count")
    if n_valid:
        print(f"post-{args.max_token_num} safe/unsafe  : {v_safe} ({100*v_safe/n_valid:.2f}%) / "
              f"{v_unsafe} ({100*v_unsafe/n_valid:.2f}%)")
    print(f"approx optimizer steps at grad_accum 8 = valid // 8 = {n_valid // 8}")


if __name__ == "__main__":
    main()
