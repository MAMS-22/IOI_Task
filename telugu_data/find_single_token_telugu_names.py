#!/usr/bin/env python3
"""
Find Telugu names that are represented by exactly one token in BLOOM.

This script does NOT modify evaluate_perplexity.py.

Two modes are supported:

1. Candidate-name mode:
   Give a text file containing Telugu names (one per line) and the script
   reports which names are exactly one BLOOM token.

   Example:
       python find_single_token_telugu_names.py \
           --names telugu_name_candidates.txt

2. Vocabulary-search mode:
   Without --names, inspect BLOOM's vocabulary and print single-token
   vocabulary entries whose decoded text contains Telugu Unicode characters.
   These are CANDIDATES, not guaranteed to be names. They should be
   manually checked before adding them to the IOI vocabulary.

The script uses tokenizer.decode([token_id]) rather than
convert_ids_to_tokens(), because BLOOM's displayed token strings can look
like UTF-8 byte fragments.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer


TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")


def load_names(path: Path) -> list[str]:
    """Load Telugu names, one per line. Blank lines and # comments are ignored."""
    if not path.exists():
        raise FileNotFoundError(f"Name file not found: {path}")

    names = []
    seen = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.split("#", 1)[0].strip()
            if name and name not in seen:
                names.append(name)
                seen.add(name)

    return names


def check_names(tokenizer, names: list[str]) -> list[dict]:
    """Return tokenization information for candidate names."""
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


def vocabulary_search(tokenizer) -> list[dict]:
    """
    Find vocabulary entries which decode to exactly one token and contain
    Telugu Unicode characters.

    Note: these are vocabulary candidates, not necessarily human names.
    """
    candidates = []

    # get_vocab() returns token -> token_id.
    for token, token_id in tokenizer.get_vocab().items():
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        if TELUGU_RE.search(decoded):
            # Verify that the text really maps back to this one token.
            reencoded = tokenizer.encode(decoded, add_special_tokens=False)

            if len(reencoded) == 1 and reencoded[0] == token_id:
                candidates.append(
                    {
                        "token_id": token_id,
                        "decoded": decoded,
                        "token": token,
                    }
                )

    candidates.sort(key=lambda x: (x["decoded"], x["token_id"]))
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find Telugu names that are exactly one token in BLOOM, "
            "without modifying the IOI evaluation script."
        )
    )

    parser.add_argument(
        "--model",
        default="bigscience/bloom-560m",
        help="Hugging Face BLOOM model/tokenizer ID.",
    )

    parser.add_argument(
        "--names",
        type=Path,
        default=None,
        help=(
            "Optional text file containing Telugu candidate names, "
            "one name per line."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output file.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "In candidate-name mode, print all names instead of only "
            "single-token names."
        ),
    )

    args = parser.parse_args()

    print(f"[*] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    if args.names:
        names = load_names(args.names)

        if not names:
            raise RuntimeError(f"No names found in {args.names}")

        results = check_names(tokenizer, names)
        single = [x for x in results if x["single_token"]]

        print(f"\n[*] Checked {len(results)} candidate names")
        print(f"[*] Single-token names: {len(single)}")

        print("\n" + "=" * 80)
        print("SINGLE-TOKEN TELUGU NAMES")
        print("=" * 80)

        if single:
            for item in single:
                print(
                    f"{item['name']:<20} -> "
                    f"token_id={item['token_ids'][0]} -> "
                    f"{item['tokens']}"
                )
        else:
            print("None found.")

        if args.all:
            print("\n" + "=" * 80)
            print("ALL CANDIDATE NAMES")
            print("=" * 80)

            for item in results:
                status = "YES" if item["single_token"] else "NO"
                print(
                    f"{status:<3} {item['name']:<20} -> "
                    f"{item['token_count']} tokens -> {item['tokens']}"
                )

        output_data = {
            "model": args.model,
            "mode": "candidate_names",
            "checked": len(results),
            "single_token_count": len(single),
            "results": results,
        }

    else:
        print("[*] No --names file supplied.")
        print("[*] Searching BLOOM vocabulary for Telugu-looking single tokens...")
        print("[*] These are candidates, NOT automatically verified names.\n")

        candidates = vocabulary_search(tokenizer)

        print("=" * 80)
        print("TELUGU SINGLE-TOKEN VOCABULARY CANDIDATES")
        print("=" * 80)
        print(f"Found: {len(candidates)}")
        print()

        for item in candidates:
            print(
                f"{item['decoded']:<20} -> "
                f"token_id={item['token_id']} -> "
                f"{item['token']!r}"
            )

        output_data = {
            "model": args.model,
            "mode": "vocabulary_search",
            "candidate_count": len(candidates),
            "candidates": candidates,
        }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n[*] Results written to: {args.output}")


if __name__ == "__main__":
    main()
