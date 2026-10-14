
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
from transformer_lens import TransformerBridge

from activation_patch_ioi import (
    MODEL_NAME,
    DEVICE,
    SEED,
    N_PER_PATTERN,
    build_examples,
    logit_diff,
)

# Start with the later layers identified by residual patching.
LAYERS_TO_TEST = list(range(13, 24))

OUTPUT_PATH = Path(
    "english_results/bloom560m_head_activation_patching_seed42.json"
)


def run_and_cache_z(model, ids):
    """Cache attention-head outputs z for the clean prompt."""
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    with torch.inference_mode():
        logits, cache = model.run_with_cache(
            input_ids,
            names_filter=lambda name: name.endswith("attn.hook_z"),
        )

    return logits, cache


def patch_one_head(model, corrupt_ids, clean_cache, layer, head, pos):
    """Patch one head's z activation at one token position."""
    input_ids = torch.tensor(
        [corrupt_ids], dtype=torch.long, device=DEVICE
    )
    hook_name = f"blocks.{layer}.attn.hook_z"

    def patch_hook(z, hook):
        z = z.clone()
        clean_z = clean_cache[hook_name]

        if pos >= z.shape[1] or pos >= clean_z.shape[1]:
            raise IndexError(
                f"Position {pos} out of range for layer {layer}"
            )

        # z shape: [batch, sequence, heads, d_head]
        z[:, pos, head, :] = clean_z[:, pos, head, :]
        return z

    with torch.inference_mode():
        logits = model.run_with_hooks(
            input_ids,
            fwd_hooks=[(hook_name, patch_hook)],
        )

    return logits


def summarize(values):
    if not values:
        return {}

    clean = sum(x["clean"] for x in values) / len(values)
    corrupted = sum(x["corrupted"] for x in values) / len(values)
    patched = sum(x["patched"] for x in values) / len(values)

    gap = clean - corrupted
    recovery = (
        (patched - corrupted) / gap
        if abs(gap) > 1e-8 else None
    )

    return {
        "n": len(values),
        "clean_mean_logit_diff": clean,
        "corrupted_mean_logit_diff": corrupted,
        "patched_mean_logit_diff": patched,
        "delta_vs_corrupted": patched - corrupted,
        "normalized_recovery": recovery,
    }


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model = TransformerBridge.boot_transformers(
        MODEL_NAME, device=DEVICE
    )
    model.enable_compatibility_mode()
    model.eval()

    examples = build_examples(model.tokenizer)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    layers = [layer for layer in LAYERS_TO_TEST if layer < n_layers]
    print(f"Examples: {len(examples)}")
    print(f"Testing layers: {layers}")
    print(f"Heads per layer: {n_heads}")
    print(f"Total candidate heads: {len(layers) * n_heads}")

    # Cache clean activations once per example. Store only scalar scores
    # and head-level activations needed for subsequent interventions.
    prepared = []
    baseline = defaultdict(list)

    for i, ex in enumerate(examples, 1):
        clean_logits, clean_cache = run_and_cache_z(
            model, ex["clean_ids"]
        )

        corrupt_ids = torch.tensor(
            [ex["corrupt_ids"]], dtype=torch.long, device=DEVICE
        )
        with torch.inference_mode():
            corrupt_logits = model(corrupt_ids)

        clean_score = logit_diff(clean_logits, ex)
        corrupt_score = logit_diff(corrupt_logits, ex)

        baseline[ex["pattern"]].append({
            "clean": clean_score,
            "corrupted": corrupt_score,
        })

        prepared.append({
            "example": ex,
            "clean_cache": clean_cache,
            "clean_score": clean_score,
            "corrupted_score": corrupt_score,
        })

        if i % 5 == 0:
            print(f"Cached {i}/{len(examples)} examples")

    results = {
        "model": MODEL_NAME,
        "seed": SEED,
        "n_per_pattern": N_PER_PATTERN,
        "layers_tested": layers,
        "heads_per_layer": n_heads,
        "patch_site": "attn.hook_z at final query position",
        "corruption": (
            "Replace the second occurrence of B with A "
            "in the action-subject position."
        ),
        "baseline": {},
        "heads": {},
    }

    for pattern in ("ABBA", "BABA"):
        rows = baseline[pattern]
        results["baseline"][pattern] = {
            "n": len(rows),
            "clean_mean_logit_diff": sum(
                x["clean"] for x in rows
            ) / len(rows),
            "corrupted_mean_logit_diff": sum(
                x["corrupted"] for x in rows
            ) / len(rows),
        }

    print("\nBaseline:")
    print(json.dumps(results["baseline"], indent=2))

    # Test one head at a time.
    for layer in layers:
        for head in range(n_heads):
            head_key = f"layer_{layer}_head_{head}"
            by_pattern = defaultdict(list)

            print(f"\nPatching {head_key}")

            for i, item in enumerate(prepared, 1):
                ex = item["example"]
                query_pos = ex["positions"]["query"]

                patched_logits = patch_one_head(
                    model,
                    ex["corrupt_ids"],
                    item["clean_cache"],
                    layer,
                    head,
                    query_pos,
                )

                patched_score = logit_diff(patched_logits, ex)

                by_pattern[ex["pattern"]].append({
                    "clean": item["clean_score"],
                    "corrupted": item["corrupted_score"],
                    "patched": patched_score,
                })

                del patched_logits

            results["heads"][head_key] = {
                pattern: summarize(rows)
                for pattern, rows in by_pattern.items()
            }

            for pattern, stats in results["heads"][head_key].items():
                recovery = stats["normalized_recovery"]
                recovery_text = (
                    f"{recovery:+.3f}"
                    if recovery is not None else "undefined"
                )
                print(
                    f"  {pattern}: recovery={recovery_text}, "
                    f"delta={stats['delta_vs_corrupted']:+.3f}"
                )

            # Save after every head so progress is not lost.
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    # Rank heads separately by pattern and by average recovery.
    for pattern in ("ABBA", "BABA"):
        ranked = []
        for head_key, per_pattern in results["heads"].items():
            stats = per_pattern.get(pattern, {})
            recovery = stats.get("normalized_recovery")
            if recovery is not None:
                ranked.append((recovery, head_key, stats))

        ranked.sort(reverse=True)

        print(f"\nTop heads for {pattern}:")
        for recovery, head_key, stats in ranked[:15]:
            print(
                f"  {head_key:18s} recovery={recovery:+.3f} "
                f"delta={stats['delta_vs_corrupted']:+.3f}"
            )

    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
