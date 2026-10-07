#!/usr/bin/env python3
"""
Telugu IOI Perplexity Evaluation

Evaluates an already-generated Telugu IOI minimal-pair corpus using:
  - BLOOM-560m
  - OLMo-1B

The corpus must contain matched "swapped" and "canonical" sentences.
No sentence generation or mock/simulated evaluation is performed here.

Expected corpus format:
{
  "language": "Telugu",
  "count": 50,
  "pairs": [
    {
      "id": 1,
      "template_id": 1,
      "A": "...",
      "B": "...",
      "place": "...",
      "object": "...",
      "initial_order": "A-B",
      "flipped": true,
      "swapped": "...",
      "canonical": "..."
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "PyTorch and Transformers are required. "
        "Install them in your ioi-ppl environment before running this script."
    ) from exc


# =============================================================================
# 1. Data loading and validation
# =============================================================================

def load_corpus(path: str) -> List[Dict[str, Any]]:
    """Load and validate the already-generated minimal-pair corpus."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "pairs" not in data:
        raise ValueError(
            "Corpus must be a JSON object containing a top-level 'pairs' field."
        )

    pairs = data["pairs"]

    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Corpus 'pairs' must be a non-empty list.")

    required = {"id", "swapped", "canonical"}
    for i, pair in enumerate(pairs):
        missing = required - set(pair)
        if missing:
            raise ValueError(
                f"Pair {i} is missing required fields: {sorted(missing)}"
            )

        if not isinstance(pair["swapped"], str) or not pair["swapped"].strip():
            raise ValueError(f"Pair {i} has an invalid swapped sentence.")

        if not isinstance(pair["canonical"], str) or not pair["canonical"].strip():
            raise ValueError(f"Pair {i} has an invalid canonical sentence.")

    return pairs


# =============================================================================
# 2. Perplexity computation
# =============================================================================

def sentence_perplexity(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int = 512,
) -> Tuple[float, float, int]:
    """
    Compute sentence-level perplexity.

    Returns:
        (perplexity, mean_loss, number_of_predicted_tokens)

    The loss is the causal-LM next-token loss returned by the model.
    """
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    input_ids = encodings["input_ids"].to(device)

    if input_ids.shape[1] < 2:
        raise ValueError(
            f"Sentence contains fewer than 2 tokens after tokenization: {text!r}"
        )

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)

    loss = float(outputs.loss.item())
    num_tokens = input_ids.shape[1] - 1

    # Guard against numerical overflow.
    if loss > 700:
        ppl = float("inf")
    else:
        ppl = math.exp(loss)

    return ppl, loss, num_tokens


def evaluate_sentences(
    model,
    tokenizer,
    sentences: List[str],
    device: torch.device,
    max_length: int = 512,
) -> Dict[str, Any]:
    """
    Evaluate a list of sentences and calculate:
      - corpus PPL: token-weighted
      - mean sentence PPL
      - standard deviation of sentence PPL
      - individual PPL/loss/token counts
    """
    records = []
    total_negative_log_likelihood = 0.0
    total_tokens = 0

    for sentence_id, text in enumerate(sentences, start=1):
        ppl, loss, num_tokens = sentence_perplexity(
            model=model,
            tokenizer=tokenizer,
            text=text,
            device=device,
            max_length=max_length,
        )

        records.append({
            "sentence_index": sentence_id,
            "perplexity": ppl,
            "loss": loss,
            "predicted_tokens": num_tokens,
        })

        total_negative_log_likelihood += loss * num_tokens
        total_tokens += num_tokens

    corpus_loss = (
        total_negative_log_likelihood / total_tokens
        if total_tokens > 0
        else float("nan")
    )

    corpus_ppl = (
        math.exp(corpus_loss)
        if math.isfinite(corpus_loss) and corpus_loss <= 700
        else float("inf")
    )

    sentence_ppls = [r["perplexity"] for r in records]
    mean_ppl = sum(sentence_ppls) / len(sentence_ppls)

    variance = sum(
        (ppl - mean_ppl) ** 2 for ppl in sentence_ppls
    ) / len(sentence_ppls)

    std_ppl = math.sqrt(variance)

    return {
        "corpus_perplexity": corpus_ppl,
        "corpus_mean_loss": corpus_loss,
        "total_predicted_tokens": total_tokens,
        "mean_sentence_perplexity": mean_ppl,
        "std_sentence_perplexity": std_ppl,
        "individual": records,
    }


# =============================================================================
# 3. Pairwise analysis
# =============================================================================

def analyze_pairs(
    pairs: List[Dict[str, Any]],
    swapped_results: Dict[str, Any],
    canonical_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Analyze the matched swapped/canonical pairs.

    Positive delta means:
        swapped PPL > canonical PPL

    Also reports:
      - absolute PPL difference
      - relative difference
      - log-PPL difference
      - win counts
    """
    swapped_individual = swapped_results["individual"]
    canonical_individual = canonical_results["individual"]

    if len(swapped_individual) != len(pairs):
        raise ValueError("Swapped result count does not match corpus pair count.")

    if len(canonical_individual) != len(pairs):
        raise ValueError("Canonical result count does not match corpus pair count.")

    pair_results = []

    swapped_wins = 0
    canonical_wins = 0
    ties = 0

    for i, pair in enumerate(pairs):
        sw = swapped_individual[i]
        can = canonical_individual[i]

        sw_ppl = sw["perplexity"]
        can_ppl = can["perplexity"]

        delta = sw_ppl - can_ppl
        ratio = sw_ppl / can_ppl if can_ppl > 0 else float("nan")

        sw_loss = sw["loss"]
        can_loss = can["loss"]

        log_ppl_delta = sw_loss - can_loss

        if sw_ppl > can_ppl:
            swapped_wins += 1
        elif can_ppl > sw_ppl:
            canonical_wins += 1
        else:
            ties += 1

        pair_results.append({
            "id": pair["id"],
            "template_id": pair.get("template_id"),
            "A": pair.get("A"),
            "B": pair.get("B"),
            "initial_order": pair.get("initial_order"),
            "flipped": pair.get("flipped"),
            "swapped_perplexity": sw_ppl,
            "canonical_perplexity": can_ppl,
            "ppl_delta_swapped_minus_canonical": delta,
            "ppl_ratio_swapped_over_canonical": ratio,
            "loss_delta_swapped_minus_canonical": log_ppl_delta,
            "swapped_tokens": sw["predicted_tokens"],
            "canonical_tokens": can["predicted_tokens"],
        })

    deltas = [
        r["ppl_delta_swapped_minus_canonical"]
        for r in pair_results
    ]

    ratios = [
        r["ppl_ratio_swapped_over_canonical"]
        for r in pair_results
        if math.isfinite(r["ppl_ratio_swapped_over_canonical"])
    ]

    loss_deltas = [
        r["loss_delta_swapped_minus_canonical"]
        for r in pair_results
    ]

    mean_delta = sum(deltas) / len(deltas)
    mean_ratio = sum(ratios) / len(ratios)
    mean_loss_delta = sum(loss_deltas) / len(loss_deltas)

    variance_delta = sum(
        (x - mean_delta) ** 2 for x in deltas
    ) / len(deltas)

    return {
        "mean_pairwise_ppl_delta": mean_delta,
        "std_pairwise_ppl_delta": math.sqrt(variance_delta),
        "mean_pairwise_ppl_ratio": mean_ratio,
        "mean_pairwise_loss_delta": mean_loss_delta,
        "swapped_higher_ppl_count": swapped_wins,
        "canonical_higher_ppl_count": canonical_wins,
        "ties": ties,
        "swapped_higher_ppl_fraction": swapped_wins / len(pair_results),
        "pair_results": pair_results,
    }


# =============================================================================
# 4. Model loading
# =============================================================================

def load_model(
    model_id: str,
    device: torch.device,
):
    """Load tokenizer and causal language model."""
    print(f"\n[*] Loading {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)

    model.eval()

    print(f"    Loaded on {device}")
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return tokenizer, model


# =============================================================================
# 5. Main evaluation
# =============================================================================

def run_evaluation(
    corpus_path: str,
    output_path: str,
    bloom_model_id: str = "bigscience/bloom-560m",
    olmo_model_id: str = "allenai/OLMo-1B-hf",
    device: str | None = None,
    max_length: int = 512,
):
    pairs = load_corpus(corpus_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_obj = torch.device(device)

    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )

    print("=" * 75)
    print("TELUGU IOI PERPLEXITY EVALUATION")
    print("=" * 75)
    print(f"Corpus: {corpus_path}")
    print(f"Pairs: {len(pairs)}")
    print(f"Device: {device_obj}")
    print(f"Max sequence length: {max_length}")

    results = {
        "corpus": {
            "path": str(corpus_path),
            "num_pairs": len(pairs),
        },
        "evaluation": {
            "device": str(device_obj),
            "max_length": max_length,
            "models": {},
        },
    }

    sentences_swapped = [pair["swapped"] for pair in pairs]
    sentences_canonical = [pair["canonical"] for pair in pairs]

    models = [
        ("BLOOM", bloom_model_id),
        ("OLMo", olmo_model_id),
    ]

    for model_name, model_id in models:
        print("\n" + "=" * 75)
        print(f"MODEL: {model_name}")
        print("=" * 75)

        tokenizer, model = load_model(model_id, device_obj)

        print(f"\n[*] Evaluating swapped sentences ({len(pairs)})...")
        swapped_results = evaluate_sentences(
            model,
            tokenizer,
            sentences_swapped,
            device_obj,
            max_length,
        )

        print(
            f"    Corpus PPL: {swapped_results['corpus_perplexity']:.4f}"
            f" | Mean sentence PPL: "
            f"{swapped_results['mean_sentence_perplexity']:.4f}"
            f" ± {swapped_results['std_sentence_perplexity']:.4f}"
        )

        print(f"\n[*] Evaluating canonical sentences ({len(pairs)})...")
        canonical_results = evaluate_sentences(
            model,
            tokenizer,
            sentences_canonical,
            device_obj,
            max_length,
        )

        print(
            f"    Corpus PPL: {canonical_results['corpus_perplexity']:.4f}"
            f" | Mean sentence PPL: "
            f"{canonical_results['mean_sentence_perplexity']:.4f}"
            f" ± {canonical_results['std_sentence_perplexity']:.4f}"
        )

        pair_analysis = analyze_pairs(
            pairs,
            swapped_results,
            canonical_results,
        )

        corpus_delta = (
            swapped_results["corpus_perplexity"]
            - canonical_results["corpus_perplexity"]
        )

        corpus_ratio = (
            swapped_results["corpus_perplexity"]
            / canonical_results["corpus_perplexity"]
            if canonical_results["corpus_perplexity"] > 0
            else float("nan")
        )

        print("\n[*] IOI comparison:")
        print(f"    Corpus PPL delta (Sw - Can): {corpus_delta:+.4f}")
        print(f"    Corpus PPL ratio (Sw / Can): {corpus_ratio:.4f}")
        print(
            f"    Mean pairwise PPL delta: "
            f"{pair_analysis['mean_pairwise_ppl_delta']:+.4f}"
        )
        print(
            f"    Mean pairwise loss delta: "
            f"{pair_analysis['mean_pairwise_loss_delta']:+.6f}"
        )
        print(
            f"    Swapped > Canonical: "
            f"{pair_analysis['swapped_higher_ppl_count']}/{len(pairs)}"
        )
        print(
            f"    Canonical > Swapped: "
            f"{pair_analysis['canonical_higher_ppl_count']}/{len(pairs)}"
        )
        print(f"    Ties: {pair_analysis['ties']}")

        results["evaluation"]["models"][model_name] = {
            "model_id": model_id,
            "swapped": swapped_results,
            "canonical": canonical_results,
            "corpus_ppl_delta_swapped_minus_canonical": corpus_delta,
            "corpus_ppl_ratio_swapped_over_canonical": corpus_ratio,
            "pairwise_analysis": pair_analysis,
        }

        del model
        del tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 75)
    print(f"[*] Results saved to: {output}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Telugu IOI perplexity on an existing corpus."
    )

    parser.add_argument(
        "--corpus",
        type=str,
        default="ioi_corpora/telugu_corpus_50.json",
        help="Path to the generated Telugu IOI corpus.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="ioi_corpora/telugu_perplexity_results.json",
        help="Path for the evaluation results JSON.",
    )

    parser.add_argument(
        "--bloom_model",
        type=str,
        default="bigscience/bloom-560m",
        help="Hugging Face model ID for BLOOM.",
    )

    parser.add_argument(
        "--olmo_model",
        type=str,
        default="allenai/OLMo-1B-hf",
        help="Hugging Face model ID for OLMo.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device, e.g. cuda or cpu. Defaults to CUDA when available.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum tokenized sequence length.",
    )

    args = parser.parse_args()

    run_evaluation(
        corpus_path=args.corpus,
        output_path=args.output,
        bloom_model_id=args.bloom_model,
        olmo_model_id=args.olmo_model,
        device=args.device,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
