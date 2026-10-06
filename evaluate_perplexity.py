"""
IOI multilingual perplexity evaluation.

External data:
  templates/hindi_templates.json
  templates/telugu_templates.json
  vocab/hindi_vocab.json
  vocab/telugu_vocab.json

The script:
1. Loads templates/vocabulary from JSON.
2. Loads the requested model tokenizers.
3. Keeps only subject/recipient names that are exactly one token in BLOOM. OLMo is a reference model and does not affect filtering.
4. Generates paired Hindi/Telugu corpora using the filtered vocabulary.
5. Optionally evaluates real model perplexity.
"""

from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
from datetime import datetime
now = datetime.now()
formatted_now = now.strftime("%Y%m%d_%H%M")
DEFAULT_OUTPUT_DIR = ROOT / f"ioi_corpora_{formatted_now}"

def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_resources() -> Tuple[dict, dict, list, list]:
    hindi_vocab = load_json(ROOT / "vocab" / "hindi_vocab.json")
    telugu_vocab = load_json(ROOT / "vocab" / "telugu_vocab.json")
    hindi_templates = load_json(ROOT / "templates" / "hindi_templates.json")
    telugu_templates = load_json(ROOT / "templates" / "telugu_templates.json")
    return hindi_vocab, telugu_vocab, hindi_templates, telugu_templates

def single_token_ids(tokenizer, text: str) -> List[int]:
    # add_special_tokens=False is essential: special BOS/EOS tokens must not
    # count toward the single-token requirement.
    return tokenizer.encode(text, add_special_tokens=False)

def filter_names_for_bloom(
    vocab: Dict[str, List[str]],
    bloom_tokenizer: Any,
    language: str,
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """Retain subject/recipient names that are exactly one token in BLOOM.

    BLOOM defines the corpus vocabulary constraint. OLMo and any other reference
    models do not affect which names are retained. Places and objects are
    unchanged.
    """
    filtered = {k: list(v) for k, v in vocab.items()}
    report = {
        "language": language,
        "filtering_model": "BLOOM",
        "rejected": {"subjects": [], "recipients": []},
        "retained": {"subjects": [], "recipients": []},
    }

    for category in ("subjects", "recipients"):
        valid = []
        for name in vocab[category]:
            ids = single_token_ids(bloom_tokenizer, name)
            info = {
                "name": name,
                "token_count": len(ids),
                "tokens": bloom_tokenizer.convert_ids_to_tokens(ids),
            }
            if len(ids) == 1:
                valid.append(name)
                report["retained"][category].append(info)
            else:
                report["rejected"][category].append(info)
        filtered[category] = valid

    if not filtered["subjects"]:
        raise RuntimeError(
            f"No single-token subject names remain for {language} in BLOOM. "
            "Revise the vocabulary or use a different BLOOM tokenizer/model."
        )
    if not filtered["recipients"]:
        raise RuntimeError(
            f"No single-token recipient names remain for {language} in BLOOM. "
            "Revise the vocabulary or use a different BLOOM tokenizer/model."
        )

    return filtered, report

def validate_templates(templates: List[dict], language: str) -> None:
    required = {"swapped", "canonical", "has_place", "has_object"}
    for i, template in enumerate(templates, start=1):
        missing = required - template.keys()
        if missing:
            raise ValueError(
                f"{language} template {i} is missing fields: {sorted(missing)}"
            )

        if "[A]" not in template["swapped"] or "[B]" not in template["swapped"]:
            raise ValueError(f"{language} swapped template {i} lacks [A]/[B].")
        if "[A]" not in template["canonical"] or "[B]" not in template["canonical"]:
            raise ValueError(f"{language} canonical template {i} lacks [A]/[B].")

def fill_slots(template: str, b_name: str, a_name: str, place: str | None, obj: str | None) -> str:
    text = template.replace("[B]", b_name).replace("[A]", a_name)

    if "[PLACE]" in text:
        if not place:
            raise ValueError("Template requires [PLACE], but no place was supplied.")
        text = text.replace("[PLACE]", place)

    if "[OBJECT]" in text:
        if not obj:
            raise ValueError("Template requires [OBJECT], but no object was supplied.")
        text = text.replace("[OBJECT]", obj)
    return text

def generate_paired_corpora(
    templates: List[dict],
    vocab: Dict[str, List[str]],
    num_sentences: int = 50,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    swapped_corpus = []
    canonical_corpus = []

    for i in range(num_sentences):
        template_id = i % len(templates)
        template = templates[template_id]
        b_name = rng.choice(vocab["subjects"])
        possible_a = [name for name in vocab["recipients"] if name != b_name]
        if not possible_a:
            raise RuntimeError("No valid recipient distinct from selected subject.")
        a_name = rng.choice(possible_a)
        place = rng.choice(vocab["places"]) if template["has_place"] else None
        obj = rng.choice(vocab["objects"]) if template["has_object"] else None

        swapped = fill_slots(
            template["swapped"], b_name, a_name, place, obj
        )
        canonical = fill_slots(
            template["canonical"], b_name, a_name, place, obj
        )

        meta = {
            "id": i + 1,
            "template_id": template_id + 1,
            "subject_B": b_name,
            "recipient_A": a_name,
            "place": place,
            "object": obj,
        }

        swapped_corpus.append({**meta, "order": "swapped", "sentence": swapped})
        canonical_corpus.append({**meta, "order": "canonical", "sentence": canonical})

    return swapped_corpus, canonical_corpus

def compute_perplexity(
    model,
    tokenizer,
    sentences: List[str],
    device: str,
    max_length: int = 512,
) -> Dict[str, Any]:

    model.eval()
    weighted_losses = []
    token_counts = []
    sentence_ppls = []

    for text in sentences:
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        input_ids = encoding.input_ids.to(device)
        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

        loss = outputs.loss.item()
        n_tokens = input_ids.shape[1] - 1
        sentence_ppls.append(math.exp(loss))
        weighted_losses.append(loss * n_tokens)
        token_counts.append(n_tokens)

    if not token_counts:
        return {
            "corpus_perplexity": float("nan"),
            "mean_sentence_perplexity": float("nan"),
            "std_sentence_perplexity": float("nan"),
            "individual_perplexities": [],
        }

    mean_ppl = sum(sentence_ppls) / len(sentence_ppls)
    variance = sum((x - mean_ppl) ** 2 for x in sentence_ppls) / len(sentence_ppls)

    return {
        "corpus_perplexity": math.exp(sum(weighted_losses) / sum(token_counts)),
        "mean_sentence_perplexity": mean_ppl,
        "std_sentence_perplexity": math.sqrt(variance),
        "individual_perplexities": sentence_ppls,
    }

def load_models(model_specs: List[Tuple[str, str]], device: str):
    tokenizers = {}
    models = {}

    for model_name, model_id in model_specs:
        print(f"\n[*] Loading tokenizer: {model_name} ({model_id})")
        tokenizers[model_name] = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )

        print(f"[*] Loading model: {model_name} ({model_id})")
        models[model_name] = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        ).to(device)

    return tokenizers, models

def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def run_evaluation(
    model_specs: List[Tuple[str, str]],
    num_sentences: int,
    output_dir: Path,
    device: str | None,
    generate_only: bool,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Check the NVIDIA driver and the PyTorch CUDA installation."
        )

    hindi_vocab, telugu_vocab, hindi_templates, telugu_templates = load_resources()

    validate_templates(hindi_templates, "Hindi")
    validate_templates(telugu_templates, "Telugu")

    print(f"[*] Execution device: {device}")

    # BLOOM is the only model that defines the single-token name requirement.
    bloom_specs = [(name, model_id) for name, model_id in model_specs if name.upper() == "BLOOM"]
    if not bloom_specs:
        raise RuntimeError(
            "A BLOOM model must be included because BLOOM defines the "
            "single-token name requirement. Example: "
            "--model BLOOM=bigscience/bloom-560m"
        )

    bloom_name, bloom_model_id = bloom_specs[0]
    print(f"[*] Loading BLOOM tokenizer for single-token validation: {bloom_model_id}")
    bloom_tokenizer = AutoTokenizer.from_pretrained(
        bloom_model_id, trust_remote_code=True
    )

    hindi_vocab, hindi_report = filter_names_for_bloom(
        hindi_vocab, bloom_tokenizer, "Hindi"
    )
    telugu_vocab, telugu_report = filter_names_for_bloom(
        telugu_vocab, bloom_tokenizer, "Telugu"
    )

    print("\n[*] BLOOM single-token vocabulary validation")
    for language, vocab in [("Hindi", hindi_vocab), ("Telugu", telugu_vocab)]:
        print(
            f"  {language}: {len(vocab['subjects'])} subjects, "
            f"{len(vocab['recipients'])} recipients retained"
        )

    save_json(output_dir / "tokenization_validation.json", {
        "filtering_model": {"name": bloom_name, "model_id": bloom_model_id},
        "hindi": hindi_report,
        "telugu": telugu_report,
    })

    hi_swapped, hi_canonical = generate_paired_corpora(
        hindi_templates, hindi_vocab, num_sentences=num_sentences
    )

    te_swapped, te_canonical = generate_paired_corpora(
        telugu_templates, telugu_vocab, num_sentences=num_sentences
    )

    corpora = {
        "hindi_swapped": hi_swapped,
        "hindi_canonical": hi_canonical,
        "telugu_swapped": te_swapped,
        "telugu_canonical": te_canonical,
    }

    for name, data in corpora.items():
        save_json(output_dir / f"{name}_{num_sentences}.json", data)

    save_json(output_dir / f"ioi_multilingual_{num_sentences}.json", corpora)

    if generate_only:
        print("\n[*] --generate_only specified. Corpus generation complete.")
        return

    tokenizers, models = load_models(model_specs, device)
    corpus_sentences = {
        key: [item["sentence"] for item in value]
        for key, value in corpora.items()
    }

    results = {}

    for model_name, _ in model_specs:
        model = models[model_name]
        tokenizer = tokenizers[model_name]
        results[model_name] = {}

        print(f"\n{'=' * 65}")
        print(f"[*] Evaluating {model_name}")
        print("=" * 65)

        for corpus_name, sentences in corpus_sentences.items():
            print(f"  Evaluating {corpus_name} ({len(sentences)} sentences)...")
            result = compute_perplexity(model, tokenizer, sentences, device)
            results[model_name][corpus_name] = result
            print(
                f"    -> Corpus PPL: {result['corpus_perplexity']:.2f} | "
                f"Mean Sent PPL: {result['mean_sentence_perplexity']:.2f} "
                f"± {result['std_sentence_perplexity']:.2f}"
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "#" * 75)
    print("                      RESULTS SUMMARY")
    print("#" * 75)

    for model_name, model_results in results.items():
        print(f"\n>>> Model: {model_name}")
        print(
            f"{'Language':<10} | {'Condition':<12} | {'Corpus PPL':<12} | "
            f"{'Mean Sent PPL':<15} | {'Delta (Sw - Can)':<16} | "
            f"{'Ratio (Sw/Can)':<14}"
        )
        print("-" * 88)

        for language in ("hindi", "telugu"):
            sw_key = f"{language}_swapped"
            can_key = f"{language}_canonical"

            if sw_key not in model_results or can_key not in model_results:
                continue

            sw = model_results[sw_key]
            can = model_results[can_key]
            delta = sw["corpus_perplexity"] - can["corpus_perplexity"]
            ratio = sw["corpus_perplexity"] / can["corpus_perplexity"]

            print(
                f"{language.capitalize():<10} | {'Swapped':<12} | "
                f"{sw['corpus_perplexity']:<12.2f} | "
                f"{sw['mean_sentence_perplexity']:<15.2f} | {'-':<16} | {'-':<14}"
            )

            print(
                f"{'':<10} | {'Canonical':<12} | "
                f"{can['corpus_perplexity']:<12.2f} | "
                f"{can['mean_sentence_perplexity']:<15.2f} | "
                f"{delta:<+16.2f} | {ratio:<14.3f}"
            )
            print("-" * 88)

    save_json(output_dir / "perplexity_results.json", results)
    print(f"\n[*] Full results written to: {output_dir / 'perplexity_results.json'}")

def parse_model_specs(args) -> List[Tuple[str, str]]:
    specs = []
    for item in args.model:
        if "=" not in item:
            raise ValueError(
                f"Invalid --model '{item}'. Use NAME=HUGGINGFACE_MODEL_ID."
            )
        name, model_id = item.split("=", 1)
        specs.append((name, model_id))
    return specs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and evaluate multilingual IOI corpora."
    )

    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model specification NAME=HUGGINGFACE_MODEL_ID. "
            "Can be supplied multiple times. Default: BLOOM=bigscience/bloom-560m"
        ),
    )

    parser.add_argument("--num_sentences", type=int, default=50)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--generate_only",
        action="store_true",
        help="Validate tokenization and generate corpora without model perplexity evaluation.",
    )

    args = parser.parse_args()
    if args.model is None:
        args.model = ["BLOOM=bigscience/bloom-560m"]

    run_evaluation(
        model_specs=parse_model_specs(args),
        num_sentences=args.num_sentences,
        output_dir=args.output_dir,
        device=args.device,
        generate_only=args.generate_only,
    )