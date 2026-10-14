# English IOI Data

Reproduces the **English part** of the Indirect Object Identification (IOI)
task from:

> Zhang et al. (2025) *"The Same But Different: Structural Similarities and
> Differences in Multilingual Language Modeling"* — ICLR 2025

which builds directly on:

> Wang et al. (2023) *"Interpretability in the Wild: a Circuit for Indirect
> Object Identification in GPT-2 Small"*

---

## Overview

The **IOI task** tests whether a language model can correctly identify the
indirect object in a sentence such as:

```
Then, Matthew and Andrew went to the gym. Matthew gave a note to [BLANK]
```

The model should predict **Andrew** (the IO, appearing once) rather than
**Matthew** (the Subject, appearing twice / duplicated).

---

## Files

| File | Description |
|------|-------------|
| `english_templates.json` | 15 BABA templates from Wang et al. (2023) |
| `english_vocab.json` | Names (single-token in GPT-2/BLOOM), places, objects |
| `generate_english_ioi.py` | Corpus generator script |
| `evaluate_english_ioi.py` | Evaluates accuracy & zero-rank rate |
| `english_corpus_200_both_*.json` | 200-item corpus (canonical + swapped) |
| `english_corpus_50_both_*.json` | 50-item corpus (for information flow routes) |

---

## Template Notation

Each template uses:

- `[B]` — **Subject** (giver; appears **twice** in sentence — duplicated name)
- `[A]` — **Indirect Object** (recipient; appears **once** — correct answer)
- `[PLACE]` — Location filler
- `[OBJECT]` — Object filler

The **canonical** condition has `[B]` first in the opening clause (normal order).
The **swapped** condition has `[A]` first (flipped control / ABBA order).

In both conditions, the model is asked to predict `[A]` at the final position.

---

## Usage

### 1. Generate Corpus

```bash
# 200 items with canonical + swapped pairs (default)
python generate_english_ioi.py --count 200 --type both --seed 42

# 50 items for information flow route analysis
python generate_english_ioi.py --count 50 --type both --seed 42

# Canonical only
python generate_english_ioi.py --count 200 --type canonical --seed 42
```

### 2. Evaluate Models

```bash
# Evaluate both GPT-2 small and BLOOM-560M
python evaluate_english_ioi.py --corpus english_corpus_200_both_*.json

# GPT-2 only, on CPU
python evaluate_english_ioi.py \
    --corpus english_corpus_200_both_*.json \
    --models gpt2 \
    --device cpu

# BLOOM only
python evaluate_english_ioi.py \
    --corpus english_corpus_200_both_*.json \
    --models bloom
```

### 3. Expected Results (Zhang et al. 2025, Table 1)

| Model | Accuracy | Zero-Rank Rate |
|-------|----------|---------------|
| GPT2-small | **99.5%** | **97.5%** |
| BLOOM-560M | **100%** | **96.0%** |

---

## Requirements

```
transformers>=4.36.0
torch>=2.0.0
```

---

## Relation to Other Languages in This Workspace

| Directory | Language | Model |
|-----------|----------|-------|
| `english_data/` | English | GPT2-small, BLOOM-560M |
| `hindi_data/` | Hindi | BLOOM-560M, OLMo |
| `telugu_data/` | Telugu | BLOOM-560M |

All follow the same template/vocabulary/generator/evaluator structure.
