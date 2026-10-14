
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
from transformer_lens import TransformerBridge


MODEL_NAME = "bigscience/bloom-560m"
SEED = 42
N_PER_PATTERN = 100
DEVICE = "cuda"

# Candidate heads: (layer, head)
CANDIDATE_HEADS = [
    (17, 9),
    (19, 8),
    (19, 15),
    (21, 6),
    (21, 8),
    (22, 10),
]

BABA_TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

# Candidate pools. The script filters names to single-token forms.
NAME_CANDIDATES = [
    "John", "Mary", "Susan", "David", "Michael", "Robert",
    "James", "William", "Linda", "Patricia", "Jennifer",
    "Elizabeth", "Barbara", "Thomas", "Charles", "Daniel",
    "Matthew", "Anthony", "Mark", "Donald", "Steven",
    "Paul", "Andrew", "Joshua", "Kenneth", "Kevin",
    "Brian", "George", "Edward", "Ronald", "Timothy",
    "Jason", "Jeffrey", "Ryan", "Jacob", "Gary",
    "Nicholas", "Eric", "Jonathan", "Stephen", "Larry",
    "Justin", "Scott", "Brandon", "Benjamin", "Samuel",
    "Gregory", "Frank", "Alexander", "Raymond",
]

PLACES = [
    "store", "office", "garden", "school", "restaurant",
    "station", "library", "museum", "park", "hospital",
    "hotel", "market", "airport", "theater", "kitchen",
]

OBJECTS = [
    "drink", "book", "letter", "gift", "flower",
    "apple", "key", "ticket", "bag", "card",
    "box", "phone", "bottle", "pen", "cake",
]


def get_name_spans(text, name, offsets):
    """Return token indices overlapping every occurrence of name."""
    char_starts = []
    start = 0

    while True:
        pos = text.find(name, start)
        if pos == -1:
            break

        # Require the occurrence to be a whole name, not part of a word.
        end = pos + len(name)
        left_ok = pos == 0 or not (
            text[pos - 1].isalnum() or text[pos - 1] == "_"
        )
        right_ok = end == len(text) or not (
            text[end].isalnum() or text[end] == "_"
        )

        if left_ok and right_ok:
            char_starts.append((pos, end))

        start = pos + 1

    spans = []
    for char_start, char_end in char_starts:
        indices = [
            i for i, (s, e) in enumerate(offsets)
            if e > s and s < char_end and e > char_start
        ]
        if indices:
            spans.append(indices)

    return spans


def build_examples(tokenizer):
    rng = random.Random(SEED)

    # Filter to names that are one token when preceded by a space.
    valid_names = []
    for name in NAME_CANDIDATES:
        ids = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(ids) == 1:
            valid_names.append(name)

    if len(valid_names) < 2:
        raise RuntimeError("Fewer than two single-token names were found.")

    print(f"Single-token names available: {len(valid_names)}")

    examples = []

    for pattern in ("ABBA", "BABA"):
        for i in range(N_PER_PATTERN):
            template = BABA_TEMPLATES[i % len(BABA_TEMPLATES)]

            if pattern == "ABBA":
                template = template.replace(
                    "[B] and [A]", "[A] and [B]", 1
                )

            a_name, b_name = rng.sample(valid_names, 2)

            text = (
                template
                .replace("[A]", a_name)
                .replace("[B]", b_name)
                .replace("[PLACE]", rng.choice(PLACES))
                .replace("[OBJECT]", rng.choice(OBJECTS))
            )

            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )

            ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]

            a_spans = get_name_spans(text, a_name, offsets)
            b_spans = get_name_spans(text, b_name, offsets)

            # All templates end with A as the expected answer.
            if not a_spans or not b_spans:
                raise RuntimeError(f"Could not locate names in: {text}")

            # Name tokens must be single-token spans.
            if any(len(span) != 1 for span in a_spans + b_spans):
                raise RuntimeError(
                    f"Name was not one token in context: {text}"
                )

            # Tokenizer indices exclude BOS; the model input includes it.
            bos_id = tokenizer.bos_token_id
            if bos_id is None:
                raise RuntimeError("Tokenizer has no BOS token.")

            input_ids = [bos_id] + ids
            a_first_pos = a_spans[0][0] + 1
            b_first_pos = b_spans[0][0] + 1
            answer_pos = a_spans[-1][0] + 1
            query_pos = answer_pos - 1

            examples.append({
                "pattern": pattern,
                "text": text,
                "input_ids": input_ids,
                "query_pos": query_pos,
                "a_first_pos": a_first_pos,
                "b_first_pos": b_first_pos,
            })

    return examples


def evaluate_logits(logits, example):
    """Score correct A against incorrect B at the next-token position."""
    q = example["query_pos"]
    a_id = example["input_ids"][example["a_first_pos"]]
    b_id = example["input_ids"][example["b_first_pos"]]

    next_logits = logits[0, q].float()
    diff = (next_logits[a_id] - next_logits[b_id]).item()
    correct = diff > 0
    zero_rank = next_logits.argmax().item() == a_id

    return {
        "logit_diff": diff,
        "correct": int(correct),
        "zero_rank": int(zero_rank),
    }


def run_one(model, example, ablate_head=None):
    input_ids = torch.tensor(
        [example["input_ids"]], dtype=torch.long, device=DEVICE
    )

    if ablate_head is None:
        with torch.inference_mode():
            logits = model(input_ids)
    else:
        layer, head = ablate_head
        hook_name = f"blocks.{layer}.attn.hook_z"

        def zero_head(z, hook):
            # z shape: [batch, sequence, heads, d_head]
            z = z.clone()
            z[:, :, head, :] = 0
            return z

        with torch.inference_mode():
            logits = model.run_with_hooks(
                input_ids,
                fwd_hooks=[(hook_name, zero_head)],
            )

    return evaluate_logits(logits, example)


def summarize(results):
    n = len(results)
    if n == 0:
        return {}

    return {
        "n": n,
        "mean_logit_diff": sum(
            r["logit_diff"] for r in results
        ) / n,
        "accuracy": sum(r["correct"] for r in results) / n,
        "zero_rank_rate": sum(
            r["zero_rank"] for r in results
        ) / n,
    }


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)

    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model = TransformerBridge.boot_transformers(
        MODEL_NAME, device=DEVICE
    )
    model.enable_compatibility_mode()
    model.eval()

    examples = build_examples(model.tokenizer)
    print(f"Built {len(examples)} examples.")

    # Baseline pass: evaluate all examples without intervention.
    baseline_raw = []
    for i, example in enumerate(examples, 1):
        baseline_raw.append(run_one(model, example))
        if i % 20 == 0:
            print(f"Baseline: {i}/{len(examples)}")

    results = {
        "model": MODEL_NAME,
        "seed": SEED,
        "n_per_pattern": N_PER_PATTERN,
        "ablation": "zero hook_z for one head at all positions",
        "baseline": {},
        "heads": {},
    }

    for pattern in ("ABBA", "BABA"):
        pattern_results = [
            r for r, ex in zip(baseline_raw, examples)
            if ex["pattern"] == pattern
        ]
        results["baseline"][pattern] = summarize(pattern_results)

    print("\nBaseline:")
    print(json.dumps(results["baseline"], indent=2))

    # Ablate one head at a time, always using the same examples.
    for head_idx, head in enumerate(CANDIDATE_HEADS, 1):
        print(f"\nAblating layer {head[0]}, head {head[1]} "
              f"({head_idx}/{len(CANDIDATE_HEADS)})")

        ablated_raw = []
        for i, example in enumerate(examples, 1):
            ablated_raw.append(run_one(model, example, ablate_head=head))

            if i % 20 == 0:
                print(f"  Processed {i}/{len(examples)}")

        head_key = f"layer_{head[0]}_head_{head[1]}"
        results["heads"][head_key] = {}

        for pattern in ("ABBA", "BABA"):
            baseline_subset = [
                r for r, ex in zip(baseline_raw, examples)
                if ex["pattern"] == pattern
            ]
            ablated_subset = [
                r for r, ex in zip(ablated_raw, examples)
                if ex["pattern"] == pattern
            ]

            base = summarize(baseline_subset)
            abl = summarize(ablated_subset)

            results["heads"][head_key][pattern] = {
                "ablated": abl,
                "delta_logit_diff": (
                    abl["mean_logit_diff"] - base["mean_logit_diff"]
                ),
                "delta_accuracy": abl["accuracy"] - base["accuracy"],
                "delta_zero_rank_rate": (
                    abl["zero_rank_rate"] - base["zero_rank_rate"]
                ),
            }

            print(
                f"  {pattern}: "
                f"logit diff {base['mean_logit_diff']:.4f} -> "
                f"{abl['mean_logit_diff']:.4f} "
                f"(delta {abl['mean_logit_diff'] - base['mean_logit_diff']:+.4f}); "
                f"accuracy {base['accuracy']:.3f} -> {abl['accuracy']:.3f}; "
                f"zero-rank {base['zero_rank_rate']:.3f} -> "
                f"{abl['zero_rank_rate']:.3f}"
            )

        out_path = Path(
            "english_results/bloom560m_ioi_head_ablation_seed42.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Saved interim results to {out_path}")

    print("\nAll ablations completed.")
    print(f"Final results: {out_path}")


if __name__ == "__main__":
    main()
