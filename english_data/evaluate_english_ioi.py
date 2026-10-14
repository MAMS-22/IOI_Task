#!/usr/bin/env python3
"""
Evaluate English IOI task performance on GPT-2 small and BLOOM-560M.

Reproduces Table 1 (IOI Task) from Zhang et al. (2025) "The Same But Different:
Structural Similarities and Differences in Multilingual Language Modeling" (ICLR 2025),
which in turn follows the evaluation setup of Wang et al. (2023).

Metrics computed for each (model, corpus_type) combination:
  - Accuracy     : fraction of examples where P(IO) > P(S)
  - Zero-rank rate: fraction of examples where IO is the #1 predicted token

Expected results (from the paper):
  GPT2-small  EN  accuracy=99.5%, zero-rank=97.5%
  BLOOM-560M  EN  accuracy=100%,  zero-rank=96.0%

Usage:
  # Generate corpus first:
  python generate_english_ioi.py --count 200 --type both --seed 42

  # Evaluate (uses GPU if available, else CPU):
  python evaluate_english_ioi.py --corpus english_corpus_200_both_*.json

  # Specify models / device explicitly:
  python evaluate_english_ioi.py \\
      --corpus english_corpus_200_both_*.json \\
      --gpt2_model gpt2 \\
      --bloom_model bigscience/bloom-560m \\
      --device cpu
"""

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(path: str | Path) -> list:
    """Load and return the items list from a generated IOI corpus JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", data.get("pairs", []))
    if not items:
        raise ValueError(f"No items found in corpus: {path}")

    # Normalise: we need canonical sentences with IO and S fields.
    normalised = []
    for item in items:
        sentence = item.get("canonical") or item.get("sentence", "")
        io_name  = item.get("IO") or item.get("A", "")
        s_name   = item.get("S")  or item.get("B", "")
        if not sentence or not io_name or not s_name:
            continue
        normalised.append({
            "sentence": sentence,
            "IO": io_name,
            "S":  s_name,
        })
    return normalised


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str, device: torch.device):
    """Load tokenizer and causal LM, return (tokenizer, model)."""
    print(f"\n[*] Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Loaded on {device}  |  Parameters: {n_params:,}")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Token-ID helpers
# ---------------------------------------------------------------------------

def get_name_token_id(tokenizer, name: str) -> int | None:
    """
    Return the single token-id for *name*, or None if *name* maps to > 1 token.

    GPT-2 / BLOOM tokenizers prepend a leading space/▁ for words in mid-
    sentence position. We try both the bare string and the space-prefixed
    variant and return whichever produces exactly 1 token.
    """
    for candidate in (name, " " + name):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


# ---------------------------------------------------------------------------
# Per-item evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_item(
    item: dict,
    tokenizer,
    model,
    device: torch.device,
) -> dict | None:
    """
    Evaluate a single IOI item.

    The sentence is tokenized; we obtain the logit distribution at the final
    token position (i.e., the position where the model predicts what comes
    *after* the last observed token, which in the IOI templates is where the
    IO name should appear).

    Returns a dict with:
      "io_rank"  : rank of IO token (0 = top prediction)
      "s_rank"   : rank of S token
      "io_prob"  : softmax probability of IO
      "s_prob"   : softmax probability of S
      "correct"  : True if P(IO) > P(S)
      "zero_rank": True if IO is rank 0
    or None if either name is multi-token.
    """
    io_id = get_name_token_id(tokenizer, item["IO"])
    s_id  = get_name_token_id(tokenizer, item["S"])

    if io_id is None or s_id is None:
        return None  # skip multi-token names

    input_ids = tokenizer.encode(item["sentence"], return_tensors="pt").to(device)

    outputs = model(input_ids)
    # logits shape: (1, seq_len, vocab_size) — predict next token after last position
    logits = outputs.logits[0, -1, :]  # (vocab_size,)

    probs = torch.softmax(logits, dim=-1)

    # Rank: 0 = most likely.  argsort descending.
    sorted_ids = torch.argsort(logits, descending=True)
    rank_map   = {int(tok_id): rank for rank, tok_id in enumerate(sorted_ids.tolist())}

    io_rank = rank_map.get(io_id, len(rank_map))
    s_rank  = rank_map.get(s_id,  len(rank_map))

    return {
        "io_rank":   io_rank,
        "s_rank":    s_rank,
        "io_prob":   float(probs[io_id]),
        "s_prob":    float(probs[s_id]),
        "correct":   io_rank < s_rank,   # P(IO) > P(S)  ↔  logit(IO) > logit(S)
        "zero_rank": io_rank == 0,
    }


# ---------------------------------------------------------------------------
# Corpus-level evaluation
# ---------------------------------------------------------------------------

def evaluate_corpus(
    items: list,
    tokenizer,
    model,
    device: torch.device,
) -> dict:
    """Run evaluation over all items and aggregate metrics."""
    results     = []
    skipped     = 0

    for item in items:
        res = evaluate_item(item, tokenizer, model, device)
        if res is None:
            skipped += 1
            continue
        results.append(res)

    if not results:
        raise RuntimeError("No items could be evaluated (all names multi-token?).")

    n           = len(results)
    accuracy    = sum(r["correct"]   for r in results) / n
    zero_rank   = sum(r["zero_rank"] for r in results) / n

    mean_io_rank = sum(r["io_rank"] for r in results) / n
    mean_s_rank  = sum(r["s_rank"]  for r in results) / n

    return {
        "n_evaluated": n,
        "n_skipped":   skipped,
        "accuracy":    accuracy,
        "zero_rank_rate": zero_rank,
        "mean_io_rank":   mean_io_rank,
        "mean_s_rank":    mean_s_rank,
        "per_item":    results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate English IOI task on GPT-2 small and BLOOM-560M, "
            "reproducing Zhang et al. (2025) Table 1."
        )
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to the generated English IOI corpus JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path for evaluation results. Auto-named if omitted.",
    )
    parser.add_argument(
        "--gpt2_model",
        type=str,
        default="gpt2",
        help="HuggingFace model ID for GPT-2 small. (default: gpt2)",
    )
    parser.add_argument(
        "--bloom_model",
        type=str,
        default="bigscience/bloom-560m",
        help="HuggingFace model ID for BLOOM. (default: bigscience/bloom-560m)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda' or 'cpu'. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["gpt2", "bloom", "both"],
        default=["both"],
        help="Which models to evaluate. (default: both)",
    )

    args = parser.parse_args()

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load corpus
    corpus_path = Path(args.corpus)
    items = load_corpus(corpus_path)
    print(f"Corpus: {corpus_path}  |  {len(items)} items")

    # Decide which models to run
    run_models = []
    for m in args.models:
        if m in ("gpt2", "both"):
            run_models.append(("GPT2-small", args.gpt2_model))
        if m in ("bloom", "both"):
            run_models.append(("BLOOM-560M", args.bloom_model))
    # deduplicate while preserving order
    seen = set()
    unique_models = []
    for entry in run_models:
        if entry not in seen:
            seen.add(entry)
            unique_models.append(entry)
    run_models = unique_models

    all_results = {
        "corpus": {
            "path": str(corpus_path),
            "n_items": len(items),
        },
        "device": str(device),
        "models": {},
    }

    header = f"{'Model':<15} {'Accuracy':>10} {'Zero-Rank':>12} {'MeanIO-Rank':>13} {'MeanS-Rank':>11}"
    print("\n" + "=" * 65)
    print("  English IOI Evaluation  (Zhang et al. 2025 / Wang et al. 2023)")
    print("=" * 65)
    print(header)
    print("-" * 65)

    for model_label, model_id in run_models:
        tokenizer, model = load_model(model_id, device)

        res = evaluate_corpus(items, tokenizer, model, device)

        acc  = res["accuracy"]       * 100
        zrr  = res["zero_rank_rate"] * 100
        mio  = res["mean_io_rank"]
        ms   = res["mean_s_rank"]

        print(
            f"  {model_label:<13} {acc:>9.1f}%  {zrr:>10.1f}%"
            f"  {mio:>11.1f}    {ms:>9.1f}"
        )

        all_results["models"][model_label] = {
            "model_id":        model_id,
            "n_evaluated":     res["n_evaluated"],
            "n_skipped":       res["n_skipped"],
            "accuracy_pct":    round(acc, 2),
            "zero_rank_rate_pct": round(zrr, 2),
            "mean_io_rank":    round(mio, 2),
            "mean_s_rank":     round(ms, 2),
        }

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("=" * 65)
    print("\nExpected results from Zhang et al. (2025) Table 1:")
    print("  GPT2-small   EN   accuracy=99.5%  zero-rank=97.5%")
    print("  BLOOM-560M   EN   accuracy=100%   zero-rank=96.0%")

    # Save results
    if args.output is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = corpus_path.parent / f"english_ioi_eval_{ts}.json"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"\n[*] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
