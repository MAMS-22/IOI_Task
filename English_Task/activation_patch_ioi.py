
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
from transformer_lens import TransformerBridge

MODEL_NAME = "bigscience/bloom-560m"
DEVICE = "cuda"
SEED = 42
N_PER_PATTERN = 20

BABA_TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

NAMES = [
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

# Patch these semantically meaningful positions at every layer.
POSITIONS = ("first_A", "first_B", "subject_B", "query")


def char_spans(text, name):
    """Find whole-word occurrences of a name in the text."""
    spans = []
    start = 0

    while True:
        pos = text.find(name, start)
        if pos < 0:
            break

        end = pos + len(name)
        left_ok = pos == 0 or not (
            text[pos - 1].isalnum() or text[pos - 1] == "_"
        )
        right_ok = end == len(text) or not (
            text[end].isalnum() or text[end] == "_"
        )

        if left_ok and right_ok:
            spans.append((pos, end))

        start = pos + 1

    return spans


def token_spans(text, name, offsets):
    result = []
    for start, end in char_spans(text, name):
        indices = [
            i for i, (s, e) in enumerate(offsets)
            if e > s and s < end and e > start
        ]
        if indices:
            result.append(indices)
    return result


def encode_text(tokenizer, text):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    return encoded["input_ids"], encoded["offset_mapping"]


def build_examples(tokenizer):
    rng = random.Random(SEED)

    valid_names = [
        name for name in NAMES
        if len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1
    ]
    if len(valid_names) < 2:
        raise RuntimeError("Not enough single-token names.")

    if tokenizer.bos_token_id is None:
        raise RuntimeError("Tokenizer has no BOS token.")

    examples = []
    for pattern in ("ABBA", "BABA"):
        accepted = 0
        attempts = 0

        while accepted < N_PER_PATTERN:
            attempts += 1
            if attempts > N_PER_PATTERN * 20:
                print(f"\nDEBUG: pattern={pattern}")
                print(f"DEBUG: valid examples generated={len(examples)}")
                print(f"DEBUG: required examples={N_PER_PATTERN}")
                print(f"DEBUG: attempts made={attempts}")
                raise RuntimeError(
                    f"Could not generate enough valid {pattern} examples."
                )

            i = accepted
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

            # Corrupt the second occurrence of B (the subject of the action)
            # by replacing it with A. Keep the final answer target as A.
            b_chars = char_spans(text, b_name)
            if len(b_chars) < 2:
                continue

            start, end = b_chars[1]
            corrupted_text = text[:start] + a_name + text[end:]

            clean_ids, clean_offsets = encode_text(tokenizer, text)
            corrupt_ids, corrupt_offsets = encode_text(
                tokenizer, corrupted_text
            )

            # Position-aligned patching requires identical token sequence
            # lengths. Skip pairs that do not meet this requirement.
            if len(clean_ids) != len(corrupt_ids):
                continue

            a_spans = token_spans(text, a_name, clean_offsets)
            b_spans = token_spans(text, b_name, clean_offsets)

            if len(a_spans) < 2 or len(b_spans) < 2:
                continue
            if any(len(s) != 1 for s in a_spans + b_spans):
                continue

            # The final A is the answer token; score its preceding position.
            positions = {
                "first_A": a_spans[0][0] + 1,   # +1 for BOS
                "first_B": b_spans[0][0] + 1,
                "subject_B": b_spans[1][0] + 1,
                "query": a_spans[-1][0],       # after BOS is added, this points to the token before final A
            }

            # Verify the intended corruption actually changed B's subject
            # token to A at the same token position.
            if positions["subject_B"] - 1 >= len(corrupt_ids):
                continue
            if clean_ids[positions["subject_B"] - 1] == corrupt_ids[
                positions["subject_B"] - 1
            ]:
                continue

            # The intended corruption must replace B's action-subject
            # token with A's token, while preserving the sequence length.
            subject_idx = b_spans[1][0]
            expected_a_id = clean_ids[a_spans[0][0]]

            if corrupt_ids[subject_idx] != expected_a_id:
                continue

            if len(clean_ids) != len(corrupt_ids):
                continue

            bos = tokenizer.bos_token_id
            examples.append({
                "pattern": pattern,
                "text": text,
                "corrupted_text": corrupted_text,
                "clean_ids": [bos] + clean_ids,
                "corrupt_ids": [bos] + corrupt_ids,
                "positions": positions,
                "a_token_id": clean_ids[a_spans[-1][0]],
                "b_token_id": clean_ids[b_spans[0][0]],
            })
            accepted += 1

    return examples


def logit_diff(logits, example):
    q = example["positions"]["query"]
    a_id = example["a_token_id"]
    b_id = example["b_token_id"]
    return (
        logits[0, q, a_id].float() -
        logits[0, q, b_id].float()
    ).item()


def run_and_cache(model, ids):
    tensor = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.inference_mode():
        logits, cache = model.run_with_cache(
            tensor,
            names_filter=lambda name: name.endswith("hook_resid_pre"),
        )
    return logits, cache



def patch_residual(model, corrupt_ids, clean_cache, layer, pos):
    tensor = torch.tensor(
        [corrupt_ids], dtype=torch.long, device=DEVICE
    )
    hook_name = f"blocks.{layer}.hook_resid_pre"

    def patch_hook(resid, hook):
        # The callback is attached specifically to hook_resid_pre.
        # Use the explicitly selected cache key, not hook.name.
        resid = resid.clone()
        clean_resid = clean_cache[hook_name]

        if pos >= resid.shape[1] or pos >= clean_resid.shape[1]:
            raise IndexError(
                f"Position {pos} out of range: "
                f"corrupt={resid.shape[1]}, clean={clean_resid.shape[1]}"
            )

        resid[:, pos, :] = clean_resid[:, pos, :]
        return resid

    with torch.inference_mode():
        logits = model.run_with_hooks(
            tensor,
            fwd_hooks=[(hook_name, patch_hook)],
        )

    return logits



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
    print(f"Built {len(examples)} aligned clean/corrupted pairs.")

    n_layers = model.cfg.n_layers
    records = defaultdict(list)
    baseline = defaultdict(list)

    for idx, ex in enumerate(examples, 1):
        clean_logits, clean_cache = run_and_cache(model, ex["clean_ids"])
        corrupt_logits, _ = run_and_cache(model, ex["corrupt_ids"])

        clean_diff = logit_diff(clean_logits, ex)
        corrupt_diff = logit_diff(corrupt_logits, ex)

        baseline[ex["pattern"]].append({
            "clean": clean_diff,
            "corrupted": corrupt_diff,
        })

        if idx <= 3:
            print(f"\nExample {idx} [{ex['pattern']}]")
            print(" CLEAN:    ", ex["text"])
            print(" CORRUPTED:", ex["corrupted_text"])
            print(
                f" clean Δ={clean_diff:.3f}, "
                f"corrupted Δ={corrupt_diff:.3f}"
            )

        for layer in range(n_layers):
            for role in POSITIONS:
                pos = ex["positions"][role]
                patched_logits = patch_residual(
                    model,
                    ex["corrupt_ids"],
                    clean_cache,
                    layer,
                    pos,
                )
                patched_diff = logit_diff(patched_logits, ex)

                records[(ex["pattern"], role, layer)].append({
                    "clean": clean_diff,
                    "corrupted": corrupt_diff,
                    "patched": patched_diff,
                })

        del clean_logits, corrupt_logits, clean_cache

        if idx % 5 == 0:
            print(f"Processed {idx}/{len(examples)} pairs")

        if idx == 1:
            ex_ids = ex["clean_ids"]
            tokens = model.to_str_tokens(
                torch.tensor([ex_ids], device=DEVICE)
            )[0]

            print("\nToken-position diagnostic:")
            for role, pos in ex["positions"].items():
                print(f"{role:10s} pos={pos:3d} token={tokens[pos]!r}")

            print("Clean tokens:    ", tokens)

            corrupt_tokens = model.to_str_tokens(
                torch.tensor([ex["corrupt_ids"]], device=DEVICE)
            )[0]
            print("Corrupted tokens:", corrupt_tokens)

    output = {
        "model": MODEL_NAME,
        "seed": SEED,
        "n_per_pattern": N_PER_PATTERN,
        "corruption": (
            "Replace the second occurrence of B (action subject) "
            "with A; keep final A as the scored answer."
        ),
        "metric": "logit(A final token) - logit(B token)",
        "baseline": {},
        "patching": {},
    }

    for pattern in ("ABBA", "BABA"):
        rows = baseline[pattern]
        clean_mean = sum(r["clean"] for r in rows) / len(rows)
        corrupt_mean = sum(r["corrupted"] for r in rows) / len(rows)
        output["baseline"][pattern] = {
            "n": len(rows),
            "clean_mean_logit_diff": clean_mean,
            "corrupted_mean_logit_diff": corrupt_mean,
            "clean_minus_corrupted": clean_mean - corrupt_mean,
        }

    for (pattern, role, layer), rows in records.items():
        clean_mean = sum(r["clean"] for r in rows) / len(rows)
        corrupt_mean = sum(r["corrupted"] for r in rows) / len(rows)
        patched_mean = sum(r["patched"] for r in rows) / len(rows)
        denom = clean_mean - corrupt_mean

        recovery = (
            (patched_mean - corrupt_mean) / denom
            if abs(denom) > 1e-8 else None
        )

        output["patching"].setdefault(pattern, {}).setdefault(
            role, {}
        )[str(layer)] = {
            "n": len(rows),
            "clean_mean_logit_diff": clean_mean,
            "corrupted_mean_logit_diff": corrupt_mean,
            "patched_mean_logit_diff": patched_mean,
            "normalized_recovery": recovery,
        }

    out_path = Path(
        "english_results/bloom560m_activation_patching_seed42.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print("\nBaseline summary:")
    print(json.dumps(output["baseline"], indent=2))

    for pattern in ("ABBA", "BABA"):
        print(f"\n{pattern}: best patching locations by normalized recovery")
        ranking = []
        for role in POSITIONS:
            for layer, values in output["patching"][pattern][role].items():
                recovery = values["normalized_recovery"]
                if recovery is not None:
                    ranking.append((recovery, role, int(layer)))
        for recovery, role, layer in sorted(ranking, reverse=True)[:10]:
            print(
                f"  layer={layer:2d}, position={role:10s}, "
                f"recovery={recovery:+.3f}"
            )

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
