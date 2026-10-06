#!/usr/bin/env python3
"""
Generate Hindi IOI minimal-pair corpora from templates and vocabulary.

For each generated pair:
  - The same A, B, PLACE, OBJECT and template are used for both conditions.
  - "swapped" vs "canonical" comes directly from the template JSON.
  - With probability --flip-prob, the first A/B mention is flipped:
        normal: B A
        flipped: A B
    This is applied to BOTH members of the minimal pair, so it does not
    change the IOI manipulation itself.

Vocabulary format:
{
  "names": [...],
  "places": [...],
  "objects": [...]
}

Template format:
{
  "swapped": "...[A]...[B]...[PLACE]...[OBJECT]...",
  "canonical": "...",
  "has_place": true,
  "has_object": true
}
"""

import argparse
import json
import random
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_inputs(templates, vocab):
    required_vocab = {"names", "places", "objects"}
    missing = required_vocab - set(vocab)
    if missing:
        raise ValueError(f"Vocabulary is missing: {sorted(missing)}")

    for key in required_vocab:
        if not isinstance(vocab[key], list) or not vocab[key]:
            raise ValueError(f"Vocabulary field '{key}' must be a non-empty list.")

    required_template = {"swapped", "canonical", "has_place", "has_object"}
    for i, template in enumerate(templates):
        missing = required_template - set(template)
        if missing:
            raise ValueError(
                f"Template {i} is missing required fields: {sorted(missing)}"
            )

        for condition in ("swapped", "canonical"):
            text = template[condition]
            for placeholder in ("[A]", "[B]"):
                if placeholder not in text:
                    raise ValueError(
                        f"Template {i} ({condition}) does not contain {placeholder}."
                    )

            if template["has_place"] and "[PLACE]" not in text:
                raise ValueError(
                    f"Template {i} ({condition}) says has_place=true but has no [PLACE]."
                )
            if template["has_object"] and "[OBJECT]" not in text:
                raise ValueError(
                    f"Template {i} ({condition}) says has_object=true but has no [OBJECT]."
                )


def fill_template(template_text, a, b, place=None, obj=None, flip=False):
    # The original templates use B A as the first name order.
    # When flipped, replace A/B roles only in the surface-order mentions.
    #
    # Since A and B occur throughout the sentence as grammatical roles,
    # we cannot globally swap A and B: that would change who gives to whom.
    # Instead, the template itself must represent the intended roles.
    #
    # For these templates, the requested ABBA variant means changing the
    # initial coordinated mention "[B] and [A]" to "[A] and [B]" while
    # preserving every later [B]/[A] role.
    if flip:
        marker = "__INITIAL_AB_MARKER__"
        # Only alter the first occurrence of the exact coordinated phrase.
        # This preserves the grammatical B -> A relationship.
        text = template_text.replace("[B] और [A]", marker, 1)
        text = text.replace(marker, "[A] और [B]", 1)
    else:
        text = template_text

    replacements = {
        "[A]": a,
        "[B]": b,
        "[PLACE]": place if place is not None else "",
        "[OBJECT]": obj if obj is not None else "",
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)

    return text


def generate_pairs(
    templates, vocab, count, seed=None, flip_prob=0.5, generation_type="both"
):
    rng = random.Random(seed)
    names = vocab["names"]
    places = vocab["places"]
    objects = vocab["objects"]

    if len(names) < 2:
        raise ValueError("At least two distinct names are required.")
    if generation_type not in {"canonical", "swapped", "both"}:
        raise ValueError("generation_type must be 'canonical', 'swapped', or 'both'.")

    items = []
    for item_id in range(1, count + 1):
        template_index = rng.randrange(len(templates))
        template = templates[template_index]
        b = rng.choice(names)
        a = rng.choice([name for name in names if name != b])
        place = rng.choice(places) if template["has_place"] else None
        obj = rng.choice(objects) if template["has_object"] else None
        flip = rng.random() < flip_prob

        swapped = fill_template(template["swapped"], a, b, place, obj, flip=flip)
        canonical = fill_template(template["canonical"], a, b, place, obj, flip=flip)

        meta = {
            "id": item_id, "template_id": template_index,
            "A": a, "B": b, "place": place, "object": obj,
            "initial_order": "A-B" if flip else "B-A", "flipped": flip,
        }

        if generation_type == "both":
            items.append({**meta, "swapped": swapped, "canonical": canonical})
        elif generation_type == "swapped":
            items.append({**meta, "type": "swapped", "sentence": swapped})
        else:
            items.append({**meta, "type": "canonical", "sentence": canonical})

    return items


def main():
    parser = argparse.ArgumentParser(
        description="Generate Hindi IOI swapped/canonical minimal pairs."
    )
    parser.add_argument(
        "--templates",
        default="./hindi_templates.json",
        help="Path to Hindi template JSON.",
    )
    parser.add_argument(
        "--vocab",
        default="./hindi_vocab.json",
        help="Path to Hindi vocabulary JSON.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of sentences/pairs to generate.",
    )
    parser.add_argument(
        "--type",
        choices=["canonical", "swapped", "both"],
        default="both",
        help="Generate only canonical, only swapped, or both (default: both).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--flip-prob",
        type=float,
        default=0.5,
        help="Probability of changing the initial B-A order to A-B.",
    )
    
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path.",
    )

    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive.")
    if not 0.0 <= args.flip_prob <= 1.0:
        raise ValueError("--flip-prob must be between 0 and 1.")

    from datetime import datetime
    now = datetime.now()
    formatted_now = now.strftime("%Y%m%d_%H%M")

    if args.output is None:
        if args.type == "both":
            filename = f"hindi_corpus_{args.count}_both_{formatted_now}.json"
        else:
            filename = f"hindi_corpus_{args.count}_{args.type}_{formatted_now}.json"

    args.output = f"./{filename}"

    templates = load_json(args.templates)
    vocab = load_json(args.vocab)
    validate_inputs(templates, vocab)

    pairs = generate_pairs(
        templates,
        vocab,
        count=args.count,
        seed=args.seed,
        flip_prob=args.flip_prob,
        generation_type=args.type,
    )

    result = {
        "language": "Hindi",
        "count": args.count,
        "generation_type": args.type,
        "seed": args.seed,
        "flip_probability": args.flip_prob,
        "templates_file": str(args.templates),
        "vocab_file": str(args.vocab),
        "pairs": pairs,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Generated {len(pairs)} minimal pairs.")
    print(f"Output: {output_path}")

    flipped = sum(item["flipped"] for item in pairs)
    print(f"Initial A-B order: {flipped}/{len(pairs)}")
    print(f"Initial B-A order: {len(pairs) - flipped}/{len(pairs)}")

    print("\nFirst 3 generated items:")
    for item in pairs[:3]:
        print(f"\nItem {item['id']} ({item['initial_order']}):")
        if args.type == "both":
            print(f"  SWAPPED:   {item['swapped']}")
            print(f"  CANONICAL: {item['canonical']}")
        else:
            print(f"  {args.type.upper()}: {item['sentence']}")



if __name__ == "__main__":
    main()
