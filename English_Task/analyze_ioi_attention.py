
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer
from transformer_lens import TransformerBridge


MODEL_NAME = "bigscience/bloom-560m"
SEED = 42
N_PER_PATTERN = 100

TEMPLATES = {
    "ABBA": [
        "Then, [A] and [B] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
        "Then, [A] and [B] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
        "Then, [A] and [B] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
        "Then, [A] and [B] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
        "Then, [A] and [B] had a long argument, and afterwards [B] said to [A]",
        "After [A] and [B] went to the [PLACE], [B] gave a [OBJECT] to [A]",
        "When [A] and [B] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
        "When [A] and [B] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
        "While [A] and [B] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
        "While [A] and [B] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
        "After the lunch, [A] and [B] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
        "Afterwards, [A] and [B] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
        "Then, [A] and [B] had a long argument. Afterwards [B] said to [A]",
        "The [PLACE] [A] and [B] went to had a [OBJECT]. [B] gave it to [A]",
        "Friends [A] and [B] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
    ],
    "BABA": [
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
    ],
}

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


def make_examples(tokenizer):
    rng = random.Random(SEED)
    valid_names = [
        name for name in NAMES
        if len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1
    ]
    examples = []

    for pattern, templates in TEMPLATES.items():
        for i in range(N_PER_PATTERN):
            template_id = i % len(templates)
            template = templates[template_id]
            a, b = rng.sample(valid_names, 2)
            prompt = (
                template.replace("[A]", a)
                .replace("[B]", b)
                .replace("[PLACE]", rng.choice(PLACES))
                .replace("[OBJECT]", rng.choice(OBJECTS))
            )
            examples.append({
                "pattern": pattern,
                "template_id": template_id + 1,
                "prompt": prompt,
                "A": a,
                "B": b,
            })

    return examples


def token_positions(text, tokenizer, name):
    """Return token positions overlapping each occurrence of a name."""
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    spans = []
    start = 0

    while True:
        char_start = text.find(name, start)
        if char_start < 0:
            break
        char_end = char_start + len(name)

        positions = [
            i for i, (s, e) in enumerate(encoded["offset_mapping"])
            if s < char_end and e > char_start
        ]
        if positions:
            spans.append(positions)

        start = char_end

    return encoded["input_ids"], spans


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    examples = make_examples(tokenizer)

    model = TransformerBridge.boot_transformers(
        MODEL_NAME, device=device
    )
    model.enable_compatibility_mode()
    model.eval()

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Average attention mass to the first mention of A and B,
    # measured at the token immediately before the final answer A.
    sums = {
        pattern: {
            "A": torch.zeros(n_layers, n_heads),
            "B": torch.zeros(n_layers, n_heads),
        }
        for pattern in TEMPLATES
    }
    counts = defaultdict(int)
    examples_out = []

    for index, ex in enumerate(examples, 1):
        text = ex["prompt"]
        ids, a_spans = token_positions(text, tokenizer, ex["A"])
        _, b_spans = token_positions(text, tokenizer, ex["B"])

        # All supplied templates end with the answer name A.
        if not a_spans or not b_spans:
            raise RuntimeError(f"Name occurrence missing: {text}")

        # Target occurrence is the final A. The preceding position
        # predicts the answer token.
        answer_positions = a_spans[-1]
        if len(answer_positions) != 1:
            raise RuntimeError(
                f"Answer name is not one token: {text}"
            )
        answer_pos = answer_positions[0]
        query_pos = answer_pos - 1

        # BOS is prepended explicitly to match the model's configured setup.
        input_ids = torch.tensor(
            [[tokenizer.bos_token_id] + ids],
            device=device,
        )
        offset = 1
        a_first = a_spans[0]
        b_first = b_spans[0]

        with torch.inference_mode():
            _, cache = model.run_with_cache(input_ids)

        pattern_key = ex["pattern"]
        attn_a = torch.zeros(n_layers, n_heads)
        attn_b = torch.zeros(n_layers, n_heads)

        for layer in range(n_layers):
            key = f"blocks.{layer}.attn.hook_pattern"
            pattern = cache[key][0].float().cpu()
            # Shape expected: [heads, query positions, key positions].
            if pattern.ndim != 3:
                raise RuntimeError(
                    f"Unexpected attention shape {pattern.shape} at {key}"
                )

            attn_a[layer] = pattern[:, query_pos + offset, a_first[0] + offset]
            attn_b[layer] = pattern[:, query_pos + offset, b_first[0] + offset]

        sums[pattern_key]["A"] += attn_a
        sums[pattern_key]["B"] += attn_b
        counts[pattern_key] += 1

        if index <= 4:
            examples_out.append({
                "pattern": pattern_key,
                "prompt": text,
                "answer_query_token": query_pos + offset,
                "first_A_token": a_first[0] + offset,
                "first_B_token": b_first[0] + offset,
            })

        if index % 20 == 0:
            print(f"Processed {index}/{len(examples)}")

    results = {
        "model": MODEL_NAME,
        "seed": SEED,
        "n_examples": len(examples),
        "metric": (
            "Attention mass from answer-prediction position to the first "
            "token of the first occurrence of each name."
        ),
        "patterns": {},
        "top_heads_by_attention_to_A": {},
        "top_heads_by_attention_to_B": {},
        "example_positions": examples_out,
    }

    for pattern in TEMPLATES:
        n = counts[pattern]
        mean_a = sums[pattern]["A"] / n
        mean_b = sums[pattern]["B"] / n
        results["patterns"][pattern] = {
            "n": n,
            "mean_attention_to_A": mean_a.tolist(),
            "mean_attention_to_B": mean_b.tolist(),
            "mean_A_minus_B": (mean_a - mean_b).tolist(),
        }

        for label, matrix in (("A", mean_a), ("B", mean_b)):
            flat = matrix.flatten()
            values, indices = torch.topk(flat, k=10)
            results[f"top_heads_by_attention_to_{label}"][pattern] = [
                {
                    "layer": int(idx // n_heads),
                    "head": int(idx % n_heads),
                    "mean_attention": float(value),
                }
                for value, idx in zip(values, indices)
            ]

    out = Path("english_results")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "bloom560m_attention_routing_seed42.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nSaved: {path}")
    for pattern in TEMPLATES:
        print(f"\n{pattern}: top heads by attention to A")
        for head in results["top_heads_by_attention_to_A"][pattern][:5]:
            print(head)
        print(f"{pattern}: top heads by attention to B")
        for head in results["top_heads_by_attention_to_B"][pattern][:5]:
            print(head)


if __name__ == "__main__":
    main()