#!/usr/bin/env python3
"""
Generate English IOI (Indirect Object Identification) minimal-pair corpora.

Follows the setup from:
  - Wang et al. (2023) "Interpretability in the Wild: a Circuit for Indirect
    Object Identification in GPT-2 Small"
  - Zhang et al. (2025) "The Same But Different: Structural Similarities and
    Differences in Multilingual Language Modeling" (ICLR 2025)

Template notation:
  [B]      = Subject name (giver; duplicated in sentence, first mention)
  [A]      = Indirect Object name (recipient; the correct prediction target)
  [PLACE]  = Location filler
  [OBJECT] = Object filler

For each generated item:
  - "canonical"  : [B] appears before [A] in the opening clause (normal IOI)
  - "swapped"    : [A] appears before [B] in the opening clause (flipped control)

The model is given the canonical sentence up to (but not including) the final
[A] token, and should predict [A] over [B].

Usage examples:
  python generate_english_ioi.py --count 200 --type both
  python generate_english_ioi.py --count 50  --type canonical --seed 0
  python generate_english_ioi.py --count 50  --type both --flip-prob 0.0
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_inputs(templates: list, vocab: dict) -> None:
    required_vocab = {"names", "places", "objects"}
    missing_vocab = required_vocab - set(vocab)
    if missing_vocab:
        raise ValueError(f"Vocabulary is missing keys: {sorted(missing_vocab)}")

    for key in required_vocab:
        if not isinstance(vocab[key], list) or not vocab[key]:
            raise ValueError(f"Vocabulary field '{key}' must be a non-empty list.")

    required_template_keys = {"canonical", "swapped", "has_place", "has_object"}
    for i, tmpl in enumerate(templates):
        missing = required_template_keys - set(tmpl)
        if missing:
            raise ValueError(
                f"Template {i} is missing required fields: {sorted(missing)}"
            )
        for cond in ("canonical", "swapped"):
            text = tmpl[cond]
            for ph in ("[A]", "[B]"):
                if ph not in text:
                    raise ValueError(
                        f"Template {i} ({cond}) does not contain placeholder {ph}."
                    )
        if tmpl["has_place"] and "[PLACE]" not in tmpl["canonical"]:
            raise ValueError(
                f"Template {i} has has_place=true but [PLACE] not in 'canonical'."
            )
        if tmpl["has_object"] and "[OBJECT]" not in tmpl["canonical"]:
            raise ValueError(
                f"Template {i} has has_object=true but [OBJECT] not in 'canonical'."
            )


# ---------------------------------------------------------------------------
# Template filling
# ---------------------------------------------------------------------------

def fill_template(text: str, a: str, b: str, place: str | None, obj: str | None) -> str:
    """Replace all placeholders in *text* with concrete values."""
    text = text.replace("[A]", a)
    text = text.replace("[B]", b)
    if place is not None:
        text = text.replace("[PLACE]", place)
    if obj is not None:
        text = text.replace("[OBJECT]", obj)
    return text


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------

def generate_corpus(
    templates: list,
    vocab: dict,
    count: int,
    seed: int = 42,
    generation_type: str = "both",
) -> list:
    """
    Generate *count* IOI items.

    Each item uses a randomly sampled template, two distinct names (IO = [A],
    Subject = [B]), an optional place, and an optional object.

    Parameters
    ----------
    templates : list
        List of template dicts (with "canonical", "swapped", "has_place",
        "has_object" keys).
    vocab : dict
        Dict with "names", "places", "objects" lists.
    count : int
        Number of items to generate.
    seed : int
        Random seed for reproducibility.
    generation_type : str
        One of "canonical", "swapped", or "both".

    Returns
    -------
    list of dicts, each containing metadata and sentence(s).
    """
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
        tmpl_idx = rng.randrange(len(templates))
        tmpl = templates[tmpl_idx]

        # Sample two distinct names.
        # Following Wang et al. (2023): IO=[A] is the prediction target;
        # Subject=[B] is the duplicated name.
        io_name = rng.choice(names)                                  # [A]
        subj_name = rng.choice([n for n in names if n != io_name])   # [B]

        place = rng.choice(places) if tmpl["has_place"] else None
        obj   = rng.choice(objects) if tmpl["has_object"] else None

        canonical_text = fill_template(
            tmpl["canonical"], a=io_name, b=subj_name, place=place, obj=obj
        )
        swapped_text = fill_template(
            tmpl["swapped"], a=io_name, b=subj_name, place=place, obj=obj
        )

        meta = {
            "id":          item_id,
            "template_id": tmpl_idx,
            "IO":          io_name,    # indirect object (answer the model should predict)
            "S":           subj_name,  # subject (duplicated name, should be suppressed)
            "place":       place,
            "object":      obj,
        }

        if generation_type == "both":
            items.append({**meta, "canonical": canonical_text, "swapped": swapped_text})
        elif generation_type == "canonical":
            items.append({**meta, "type": "canonical", "sentence": canonical_text})
        else:  # "swapped"
            items.append({**meta, "type": "swapped", "sentence": swapped_text})

    return items


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate English IOI minimal-pair corpora following Wang et al. (2023) "
            "and Zhang et al. (2025)."
        )
    )
    parser.add_argument(
        "--templates",
        default="./english_templates.json",
        help="Path to English templates JSON. (default: ./english_templates.json)",
    )
    parser.add_argument(
        "--vocab",
        default="./english_vocab.json",
        help="Path to English vocabulary JSON. (default: ./english_vocab.json)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Number of items to generate. (default: 200)",
    )
    parser.add_argument(
        "--type",
        choices=["canonical", "swapped", "both"],
        default="both",
        help=(
            "Generate canonical only, swapped only, or both. "
            "(default: both)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. (default: 42)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Auto-generated timestamp name if omitted.",
    )

    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be a positive integer.")

    templates = load_json(args.templates)
    vocab = load_json(args.vocab)

    # Strip internal comment key if present
    vocab = {k: v for k, v in vocab.items() if not k.startswith("_")}

    validate_inputs(templates, vocab)

    items = generate_corpus(
        templates=templates,
        vocab=vocab,
        count=args.count,
        seed=args.seed,
        generation_type=args.type,
    )

    # Build output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"english_corpus_{args.count}_{args.type}_{timestamp}.json"
        output_path = Path(".") / filename
    else:
        output_path = Path(args.output)

    result = {
        "language": "English",
        "source": "Wang et al. (2023) IOI templates",
        "count": args.count,
        "generation_type": args.type,
        "seed": args.seed,
        "templates_file": str(args.templates),
        "vocab_file": str(args.vocab),
        "items": items,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Generated {len(items)} items ({args.type}).")
    print(f"Output: {output_path}")

    print("\nFirst 3 generated items:")
    for item in items[:3]:
        print(f"\nItem {item['id']} | IO={item['IO']} | S={item['S']}")
        if args.type == "both":
            print(f"  CANONICAL : {item['canonical']}")
            print(f"  SWAPPED   : {item['swapped']}")
        else:
            print(f"  {args.type.upper():<10}: {item['sentence']}")


if __name__ == "__main__":
    main()
