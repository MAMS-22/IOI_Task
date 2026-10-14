
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformer_lens import TransformerBridge


MODEL_NAME = "bigscience/bloom-560m"

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

NAMES = [
    "John", "Mary", "Michael", "Jessica", "David", "Sarah",
    "Robert", "Lisa", "William", "Jennifer", "James", "Emily",
    "Daniel", "Ashley", "Matthew", "Amanda", "Andrew", "Laura",
    "Ryan", "Rebecca", "Brian", "Michelle", "Kevin", "Rachel",
    "Thomas", "Nicole", "Steven", "Anna", "Joseph", "Emma",
]

PLACES = [
    "store", "market", "office", "garden", "school",
    "restaurant", "library", "station", "park", "beach",
]

OBJECTS = [
    "apple", "book", "drink", "flower", "gift",
    "letter", "meal", "snack", "toy", "ticket",
]


def build_templates():
    """Create ABBA counterparts by changing only the initial name order."""
    result = []

    for index, baba in enumerate(BABA_TEMPLATES):
        # Replace the first pair only. Keep the later B subject and A target.
        marker = "[B] and [A]"
        if marker not in baba:
            raise ValueError(f"Initial name pair not found: {baba}")

        abba = baba.replace(marker, "[A] and [B]", 1)

        result.append({
            "id": index + 1,
            "BABA": baba,
            "ABBA": abba,
        })

    return result


def single_token_names(tokenizer):
    valid = []

    for name in NAMES:
        ids = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(ids) == 1:
            valid.append(name)

    if len(valid) < 2:
        raise RuntimeError("Not enough single-token names.")

    return valid


def make_examples(tokenizer, n_per_pattern, seed):
    """Distribute examples as evenly as possible over all 15 templates."""
    rng = random.Random(seed)
    templates = build_templates()
    names = single_token_names(tokenizer)

    examples = {"ABBA": [], "BABA": []}

    for pattern in ("ABBA", "BABA"):
        base_count, remainder = divmod(n_per_pattern, len(templates))
        counts = [
            base_count + (i < remainder)
            for i in range(len(templates))
        ]

        seen_prompts = set()

        for template, count in zip(templates, counts):
            template_text = template[pattern]
            created = 0
            attempts = 0

            while created < count:
                attempts += 1
                if attempts > 10000:
                    raise RuntimeError(
                        f"Could not generate enough unique prompts for "
                        f"{pattern}, template {template['id']}."
                    )

                a, b = rng.sample(names, 2)
                place = rng.choice(PLACES)
                obj = rng.choice(OBJECTS)

                prompt = (
                    template_text
                    .replace("[A]", a)
                    .replace("[B]", b)
                    .replace("[PLACE]", place)
                    .replace("[OBJECT]", obj)
                )

                if prompt in seen_prompts:
                    continue

                seen_prompts.add(prompt)

                examples[pattern].append({
                    "pattern": pattern,
                    "template_id": template["id"],
                    "template": template_text,
                    "prompt": prompt,
                    "A": a,
                    "B": b,
                    "place": place,
                    "object": obj,
                })

                created += 1

    return examples, templates, names


def tokenize_example(tokenizer, example, name_ids):
    prompt_ids = tokenizer.encode(
        example["prompt"], add_special_tokens=False
    )

    # The answer name is the final token in every supplied template.
    expected_io = name_ids[example["A"]]
    expected_s = name_ids[example["B"]]

    if prompt_ids[-1] != expected_io:
        raise ValueError(
            "Target name is not tokenized as the expected final token. "
            f"Prompt: {example['prompt']!r}; "
            f"expected token ID {expected_io}, got {prompt_ids[-1]}."
        )

    bos = tokenizer.bos_token_id
    if bos is None:
        raise ValueError("Tokenizer has no BOS token; inspect tokenization.")

    return [bos] + prompt_ids, expected_io, expected_s


def evaluate_pattern(model, tokenizer, examples, device, batch_size):
    name_ids = {
        name: tokenizer.encode(" " + name, add_special_tokens=False)[0]
        for name in single_token_names(tokenizer)
    }

    prepared = []
    for ex in examples:
        ids, io_id, s_id = tokenize_example(tokenizer, ex, name_ids)
        prepared.append((ex, ids, io_id, s_id))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    per_template = defaultdict(list)
    all_results = []
    model.eval()

    for start in range(0, len(prepared), batch_size):
        batch = prepared[start:start + batch_size]
        lengths = [len(item[1]) for item in batch]
        max_len = max(lengths)

        input_ids = torch.full(
            (len(batch), max_len),
            pad_id,
            dtype=torch.long,
            device=device,
        )

        for i, (_, ids, _, _) in enumerate(batch):
            input_ids[i, :len(ids)] = torch.tensor(
                ids, dtype=torch.long, device=device
            )

        with torch.inference_mode():
            output = model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output

        for i, (ex, ids, io_id, s_id) in enumerate(batch):
            # The final prompt token is the correct answer. Predict it
            # using the logits at the preceding position.
            position = lengths[i] - 2
            next_logits = logits[i, position].float()

            io_logit = next_logits[io_id].item()
            s_logit = next_logits[s_id].item()
            top_id = next_logits.argmax().item()

            row = {
                "pattern": ex["pattern"],
                "template_id": ex["template_id"],
                "prompt": ex["prompt"],
                "correct_name": ex["A"],
                "incorrect_name": ex["B"],
                "accuracy": io_logit > s_logit,
                "zero_rank": top_id == io_id,
                "logit_difference": io_logit - s_logit,
                "top_prediction": tokenizer.decode([top_id]),
            }

            all_results.append(row)
            per_template[ex["template_id"]].append(row)

    def summarize(rows):
        n = len(rows)
        return {
            "n": n,
            "accuracy_percent": 100 * sum(r["accuracy"] for r in rows) / n,
            "zero_rank_percent": 100 * sum(r["zero_rank"] for r in rows) / n,
            "mean_logit_difference": (
                sum(r["logit_difference"] for r in rows) / n
            ),
        }

    return {
        "overall": summarize(all_results),
        "per_template": {
            str(template_id): summarize(rows)
            for template_id, rows in sorted(per_template.items())
        },
        "examples": all_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-total", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="english_results")
    args = parser.parse_args()

    if args.n_total <= 0 or args.n_total % 2 != 0:
        parser.error("--n-total must be a positive even number.")

    n_per_pattern = args.n_total // 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Total examples: {args.n_total}")
    print(f"Per pattern: {n_per_pattern}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples, templates, names = make_examples(
        tokenizer, n_per_pattern, args.seed
    )

    print(f"Single-token names available: {len(names)}")
    print(f"Templates: {len(templates)}")

    for pattern in ("ABBA", "BABA"):
        print(f"{pattern}: {len(examples[pattern])} examples")
        counts = defaultdict(int)
        for ex in examples[pattern]:
            counts[ex["template_id"]] += 1
        print("  Per-template counts:", dict(sorted(counts.items())))

    model = TransformerBridge.boot_transformers(
        MODEL_NAME, device=device
    )
    model.enable_compatibility_mode()
    model.eval()

    results = {
        "model": MODEL_NAME,
        "seed": args.seed,
        "n_total": args.n_total,
        "n_per_pattern": n_per_pattern,
        "templates": templates,
        "metrics": {},
    }

    for pattern in ("ABBA", "BABA"):
        print(f"\nEvaluating {pattern}...")
        result = evaluate_pattern(
            model,
            tokenizer,
            examples[pattern],
            device,
            args.batch_size,
        )
        results["metrics"][pattern] = result

        overall = result["overall"]
        print(f"Samples: {overall['n']}")
        print(f"Accuracy: {overall['accuracy_percent']:.2f}%")
        print(f"Zero-rank: {overall['zero_rank_percent']:.2f}%")
        print(
            "Mean logit difference: "
            f"{overall['mean_logit_difference']:.4f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"bloom560m_ioi_15templates_{args.n_total}_seed{args.seed}.json"
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()