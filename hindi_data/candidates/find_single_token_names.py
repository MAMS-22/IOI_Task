#!/usr/bin/env python3
"""
Check candidate names against the BLOOM tokenizer.

Input can be either:

1. An IOI vocabulary JSON:
       {
         "subjects": [...],
         "recipients": [...],
         "places": [...],
         "objects": [...]
       }

2. A plain text candidate list, one name per line.
   Lines beginning with # and blank lines are ignored.

The script does NOT modify the input file or evaluate_perplexity.py.

Examples:

    # Existing vocabulary JSON
    python find_single_token_names.py \
        --language hindi \
        --vocab hindi_vocab.json

    # Plain candidate-name list
    python find_single_token_names.py \
        --language hindi \
        --names hindi_name_candidates.txt

    # Save a complete JSON report
    python find_single_token_names.py \
        --language hindi \
        --names hindi_name_candidates.txt \
        --output hindi_name_tokenization.json

    # Show every candidate, not just single-token names
    python find_single_token_names.py \
        --language hindi \
        --names hindi_name_candidates.txt \
        --show-all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def load_vocab(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    required = {"subjects", "recipients"}
    missing = required - vocab.keys()

    if missing:
        raise ValueError(
            f"Vocabulary file is missing required fields: {sorted(missing)}"
        )

    return vocab


def load_candidate_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    names = []
    seen = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            # Remove comments and surrounding whitespace.
            name = line.split("#", 1)[0].strip()

            if name and name not in seen:
                names.append(name)
                seen.add(name)

    if not names:
        raise ValueError(f"No candidate names found in {path}")

    return names


def check_names(tokenizer, names: list[str]) -> list[dict]:
    results = []

    for name in names:
        ids = tokenizer.encode(name, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)

        results.append(
            {
                "name": name,
                "token_count": len(ids),
                "token_ids": ids,
                "tokens": tokens,
                "single_token": len(ids) == 1,
            }
        )

    return results


def print_results(
    language: str,
    category: str,
    results: list[dict],
) -> None:
    single = [x for x in results if x["single_token"]]

    print("\n" + "=" * 80)
    print(f"{language.upper()} — {category.upper()}")
    print("=" * 80)
    print(f"Checked: {len(results)}")
    print(f"Single-token: {len(single)}")

    if not single:
        print("\nNo single-token names found.")
        return

    print("\nSingle-token names:")

    for item in single:
        print(
            f"{item['name']:<20} -> "
            f"token_id={item['token_ids'][0]} -> "
            f"{item['tokens']}"
        )


def print_candidate_results(
    language: str,
    category: str,
    results: list[dict],
) -> None:
    single = [x for x in results if x["single_token"]]

    print("\n" + "=" * 80)
    print(f"{language.upper()} — {category.upper()} CANDIDATES")
    print("=" * 80)
    print(f"Checked: {len(results)}")
    print(f"Single-token: {len(single)}")

    print("\nSingle-token names:")

    if not single:
        print("None found.")
        return

    for item in single:
        print(
            f"{item['name']:<20} -> "
            f"token_id={item['token_ids'][0]} -> "
            f"{item['tokens']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find names that are exactly one token in BLOOM. "
            "Accepts either an IOI vocabulary JSON or a plain text "
            "candidate-name list."
        )
    )

    parser.add_argument(
        "--language",
        required=True,
        choices=["hindi", "telugu"],
        help="Language of the items being checked.",
    )

    parser.add_argument(
        "--category",
        default="names",
        help="Category for a plain-text input, e.g. names, places, objects.",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--vocab",
        type=Path,
        help=(
            "IOI vocabulary JSON containing subjects and recipients."
        ),
    )

    input_group.add_argument(
        "--names",
        type=Path,
        help=(
            "Plain text file containing candidate names, one per line."
        ),
    )

    parser.add_argument(
        "--model",
        default="bigscience/bloom-560m",
        help="Hugging Face tokenizer/model ID. Default: BLOOM-560m.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output file containing the full report.",
    )

    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print tokenization information for every candidate.",
    )

    args = parser.parse_args()

    print(f"[*] Loading tokenizer: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    if args.vocab:
        vocab = load_vocab(args.vocab)

        report = {
            "language": args.language,
            "model": args.model,
            "input_type": "vocab_json",
            "categories": {},
        }

        for category in ("subjects", "recipients"):
            results = check_names(tokenizer, vocab[category])
            report["categories"][category] = results

            print_results(
                args.language,
                category,
                results,
            )

            if args.show_all:
                print("\nAll names:")

                for item in results:
                    status = "YES" if item["single_token"] else "NO"

                    print(
                        f"{status:<3} {item['name']:<20} -> "
                        f"{item['token_count']} tokens -> "
                        f"{item['tokens']}"
                    )

    else:
        names = load_candidate_names(args.names)

        results = check_names(tokenizer, names)

        report = {
            "language": args.language,
            "model": args.model,
            "input_type": "candidate_text_list",
            "category": args.category,
            "candidate_count": len(results),
            "single_token_count": sum(
                item["single_token"] for item in results
            ),
            "results": results,
        }

        print_candidate_results(
            args.language,
            args.category,
            results,
        )

        if args.show_all:
            print("\nAll candidates:")

            for item in results:
                status = "YES" if item["single_token"] else "NO"

                print(
                    f"{status:<3} {item['name']:<20} -> "
                    f"{item['token_count']} tokens -> "
                    f"{item['tokens']}"
                )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)

        with args.output.open("w", encoding="utf-8") as f:
            json.dump(
                report,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\n[*] Full report written to: {args.output}")


if __name__ == "__main__":
    main()
