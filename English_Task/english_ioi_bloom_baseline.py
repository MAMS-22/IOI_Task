#!/usr/bin/env python3
"""Provisional English IOI benchmark for BLOOM-560M.

Caveat: the paper reports using 15 templates but does not print their exact wording.
The template bank here is a reconstruction for pipeline testing, NOT a verified copy
of the authors' templates. Do not treat its scores as an exact reproduction yet.
"""
import argparse, json, random
from datetime import datetime, timezone
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "bigscience/bloom-560m"
TEMPLATES = [
    "When {first} and {second} went to the {place}, {subject} gave a {object} to",
    "{first} and {second} were going to the {place}. {subject} handed a {object} to",
    "After {first} and {second} arrived at the {place}, {subject} passed a {object} to",
    "{first} and {second} went to the {place}, where {subject} offered a {object} to",
    "At the {place}, {first} and {second} met. {subject} gave the {object} to",
    "{first} and {second} visited the {place}. {subject} gave a {object} to",
    "While {first} and {second} were at the {place}, {subject} gave the {object} to",
    "{first} and {second} were at the {place}, and {subject} handed the {object} to",
    "Yesterday, {first} and {second} went to the {place}. {subject} gave a {object} to",
    "This morning {first} and {second} arrived at the {place}, and {subject} gave a {object} to",
    "At the {place}, {first} and {second} talked. Then {subject} gave a {object} to",
    "{subject} went with {first} and {second} to the {place}, then gave a {object} to",
    "After meeting {first} and {second} at the {place}, {subject} gave a {object} to",
    "{first} met {second} at the {place}. Later, {subject} gave a {object} to",
    "When {first} saw {second} at the {place}, {subject} decided to give a {object} to",
]
NAMES = ["Mary", "John", "Susan", "David", "Sarah", "Michael", "Jessica", "Robert", "Linda", "William", "Karen", "James", "Patricia", "Thomas", "Jennifer", "Charles", "Elizabeth", "Daniel", "Barbara", "Matthew", "Nancy", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley", "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle", "Kenneth", "Dorothy", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa", "Edward", "Deborah", "Ronald", "Stephanie", "Timothy", "Rebecca", "Jason"]
PLACES = ["store", "park", "museum", "school", "restaurant", "office", "theater"]
OBJECTS = ["book", "drink", "gift", "letter", "ticket", "apple", "sandwich"]

def token_id(tokenizer, name):
    ids = tokenizer.encode(" " + name, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None

def build_examples(tokenizer, n, seed):
    rng = random.Random(seed)
    names = [name for name in NAMES if token_id(tokenizer, name) is not None]
    if len(names) < 2:
        raise RuntimeError("Fewer than two candidate names are single tokens for this tokenizer.")
    examples = []
    for i in range(n):
        template = TEMPLATES[i % len(TEMPLATES)]
        indirect, subject = rng.sample(names, 2)
        shared = dict(subject=subject, object=rng.choice(OBJECTS), place=rng.choice(PLACES))
        for condition in ("normal", "flipped"):
            first, second = ((indirect, subject) if condition == "normal" else (subject, indirect))
            prompt = template.format(first=first, second=second, **shared)
            examples.append({"example_id": f"{i:04d}_{condition}", "template_id": i % len(TEMPLATES),
                "condition": condition, "prompt": prompt, "correct_name": indirect,
                "incorrect_name": subject, "correct_token_id": token_id(tokenizer, indirect),
                "incorrect_token_id": token_id(tokenizer, subject)})
    return examples, names

@torch.inference_mode()
def evaluate(model, tokenizer, examples, device, batch_size):
    rows = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        enc = tokenizer([x["prompt"] for x in batch], return_tensors="pt", padding=True, add_special_tokens=False)
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "Model produced NaN or Inf logits. "
                "Check model precision and input tensors."
            )
        positions = mask.sum(dim=1) - 1
        for j, ex in enumerate(batch):
            next_logits = logits[j, positions[j], :]
            correct = next_logits[ex["correct_token_id"]].item()
            incorrect = next_logits[ex["incorrect_token_id"]].item()
            rows.append({**ex, "correct_logit": correct, "incorrect_logit": incorrect,
                "correct_preferred": bool(correct > incorrect),
                "zero_rank": bool(next_logits.max().item() <= correct + 1e-7)})
    return rows

def summarize(rows):
    result = {}
    for condition in ("normal", "flipped"):
        group = [r for r in rows if r["condition"] == condition]
        result[condition] = {"n": len(group),
            "accuracy_percent": 100 * sum(r["correct_preferred"] for r in group) / len(group),
            "zero_rank_rate_percent": 100 * sum(r["zero_rank"] for r in group) / len(group)} if group else {"n": 0}
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--n-examples", type=int, default=200, help="Examples per condition (default: 200).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output-dir", default="english_results")
    args = p.parse_args()
    if args.n_examples < 1 or args.batch_size < 1: p.error("n-examples and batch-size must be positive")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    examples, valid_names = build_examples(tokenizer, args.n_examples, args.seed)
    print(f"Single-token names: {len(valid_names)}; examples: {len(examples)}")
    rows = evaluate(model, tokenizer, examples, device, args.batch_size)
    summary = summarize(rows)
    payload = {"model": args.model, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": device, "seed": args.seed, "requested_examples_per_condition": args.n_examples,
        "template_source": "provisional reconstruction; not verified as the paper's exact 15 templates",
        "metric_definitions": {"accuracy": "correct-name next-token logit > incorrect-name next-token logit",
        "zero_rank_rate": "correct-name next-token logit is tied for maximum over the full vocabulary"},
        "paper_reference_metrics_percent": {"normal_accuracy": 100.0, "flipped_accuracy": 100.0,
        "normal_zero_rank_rate": 96.0, "flipped_zero_rank_rate": 91.0}, "summary": summary, "examples": rows}
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"bloom560m_english_ioi_seed{args.seed}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for cond, s in summary.items():
        print(f"{cond}: n={s['n']}, accuracy={s['accuracy_percent']:.2f}%, zero-rank={s['zero_rank_rate_percent']:.2f}%")
    print(f"Saved results to: {out.resolve()}")
    print("NOTE: Treat this as a pipeline smoke test until the exact paper templates and sampling conventions are verified.")

if __name__ == "__main__": main()
