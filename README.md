# IOI Task

This repository contains the code, templates, vocabularies, generated corpora, and evaluation results for an **Indirect Object Identification (IOI)** task in Hindi and Telugu.

The Telugu pipeline is under active development. Vocabulary selection, gender-aware templates, corpus generation, and real-model perplexity evaluation are now implemented, but the Telugu corpus still requires linguistic quality review before being treated as final.

## Repository Structure

```text
IOI_Task/
├── evaluate_perplexity.py          # Ignored / not documented here
├── README.md
│
├── hindi_data/
│   ├── candidates/
│   │   ├── find_single_token_names.py
│   │   ├── hindi_name_candidates.txt
│   │   ├── hindi_object_candidates.txt
│   │   ├── hindi_place_candidates.txt
│   │   └── hindi_vocab_candidates.json
│   │
│   ├── hindi_templates.json
│   ├── hindi_vocab.json
│   ├── generate_hindi_ioi.py
│   ├── evaluate_hindi_perplexity.py
│   ├── hindi_corpus_*.json
│   └── hindi_perplexity_results_*.json
│
└── telugu_data/
    ├── find_single_token_telugu_names.py
    ├── telugu_name_candidates.txt
    ├── telugu_templates.json
    └── telugu_vocab.json
```

## Hindi Perplexity Evaluation

`evaluate_hindi_perplexity.py` evaluates generated Hindi IOI data using language models and computes real perplexity values.

The evaluation includes:

- Sentence-level perplexity
- Corpus-level, token-weighted perplexity
- Mean and standard deviation of sentence perplexity
- Pairwise perplexity differences
- Pairwise loss differences
- Pairwise comparisons between swapped and canonical sentences

The current evaluation setup includes:

- `bigscience/bloom-560m`
- `allenai/OLMo-1B-hf`

The generated perplexity result files are stored as:

```text
hindi_perplexity_results_*.json
```


## Hindi Perplexity Results

The Hindi IOI dataset was evaluated using two language models:

- **BLOOM-560M** (`bigscience/bloom-560m`)
- **OLMo-1B** (`allenai/OLMo-1B-hf`)

The evaluation was performed on a 50-pair Hindi IOI corpus. Each pair contains a swapped and canonical sentence, allowing the two forms to be compared directly.

### BLOOM-560M

| Metric | Swapped | Canonical |
|---|---:|---:|
| Corpus PPL | 124.8810 | 98.6223 |
| Mean sentence PPL | 151.5451 | 116.0534 |
| Sentence PPL Std. Dev. | 86.0603 | 58.9335 |

Additional pairwise results:

| Metric | Result |
|---|---:|
| PPL difference (Swapped − Canonical) | +26.2588 |
| PPL ratio (Swapped / Canonical) | 1.2663 |
| Mean pairwise PPL difference | +35.4917 |
| Mean pairwise loss difference | +0.244965 |
| Swapped > Canonical | 44 / 50 |
| Canonical > Swapped | 4 / 50 |
| Ties | 2 / 50 |

### OLMo-1B

| Metric | Swapped | Canonical |
|---|---:|---:|
| Corpus PPL | 5.2930 | 5.1088 |
| Mean sentence PPL | 5.4784 | 5.2793 |
| Sentence PPL Std. Dev. | 1.0134 | 0.9382 |

Additional pairwise results:

| Metric | Result |
|---|---:|
| PPL difference (Swapped − Canonical) | +0.1843 |
| PPL ratio (Swapped / Canonical) | 1.0361 |
| Mean pairwise PPL difference | +0.1992 |
| Mean pairwise loss difference | +0.036155 |
| Swapped > Canonical | 39 / 50 |
| Canonical > Swapped | 9 / 50 |
| Ties | 2 / 50 |

### Result Summary

Both models assign higher perplexity to the swapped sentences on average, producing the expected IOI contrast.

The effect is substantially stronger for **BLOOM-560M**, where the swapped corpus has a perplexity of `124.8810` compared with `98.6223` for the canonical corpus. The swapped form also has a higher perplexity in **44 of 50** sentence pairs.

For **OLMo-1B**, the same direction is observed but with a much smaller difference: `5.2930` versus `5.1088`, with swapped sentences having higher perplexity in **39 of 50** pairs.

These results provide an initial indication that the Hindi IOI dataset captures the intended distinction between the canonical and swapped constructions. OLMo is included as a reference model; BLOOM-560M provides the stronger signal in the current Hindi evaluation.


## Hindi Data

The Hindi pipeline currently contains the complete components required for generating and evaluating IOI data.

### Vocabulary

`hindi_vocab.json` contains the vocabulary used for generation:

- Names
- Places
- Objects

The vocabulary was selected with model tokenization in mind, particularly to ensure suitable single-token names for the BLOOM model used in evaluation.

The `candidates/` directory contains the intermediate candidate lists and the script used to identify suitable vocabulary items.

### Templates

`hindi_templates.json` contains the Hindi sentence templates used to construct IOI examples.

The templates provide:

- A canonical sentence
- A swapped sentence
- Optional place and object slots
- Information about whether a template uses a place or object

The generation process preserves the grammatical roles of the names while changing the relevant ordering needed for the IOI contrast.

### Dataset Generation

`generate_hindi_ioi.py` generates Hindi IOI corpora from the templates and vocabulary.

The generator supports three modes:

```text
canonical
swapped
both
```

For example:

```bash
python generate_hindi_ioi.py \
    --templates hindi_templates.json \
    --vocab hindi_vocab.json \
    --count 200 \
    --type both
```

The generator can also randomly flip the initial ordering of the two names. This allows both initial orders, BABA and ABBA, to occur while preserving the intended grammatical relationship in the subsequent sentence.

Generated files use names of the following form:

```text
hindi_corpus_<count>_<type>_<timestamp>.json
```

Examples:

```text
hindi_corpus_200_both_20261006_1250.json
hindi_corpus_200_swapped_20261006_1254.json
```

When `both` is selected, each item contains the matched canonical and swapped sentences. When `canonical` or `swapped` is selected, each item contains only the requested sentence type.


## Telugu Data

The Telugu portion of the repository is under active development. The current pipeline includes vocabulary verification, gender-aware templates, corpus generation, and real-model perplexity evaluation.

### Vocabulary

`telugu_vocab_verified.json` contains vocabulary verified against the BLOOM-560M tokenizer.

The current verified name vocabulary contains 19 names with explicit gender labels:

- 8 masculine names
- 11 feminine names

The vocabulary also contains the single-token places:

- `బడి`
- `కొండ`
- `సభ`

and the single-token object:

- `పత్రిక`

Gender labels are used by the generator to select the appropriate Telugu verb form for the grammatical agent.

### Templates

`telugu_templates_gender_aware.json` contains 15 Telugu IOI templates.

The templates provide:

- A canonical sentence
- A swapped sentence
- Optional place and object slots
- Gender-specific verb forms
- Information about whether a template uses a place or object

For example, masculine and feminine forms are represented separately:

```text
ఇచ్చాడు / ఇచ్చింది
చెప్పాడు / చెప్పింది
ఇవ్వాలనుకున్నాడు / ఇవ్వాలనుకుంది
నిర్ణయించుకున్నాడు / నిర్ణయించుకుంది
```

The generator preserves the grammatical role of the agent while changing the relevant ordering for the IOI contrast.

### Dataset Generation

`generate_telugu_ioi_gender_aware.py` generates Telugu IOI corpora using the gender-aware vocabulary and templates.

The generator supports:

```text
canonical
swapped
both
```

Example:

```bash
python generate_telugu_ioi_gender_aware.py \
    --templates telugu_templates_gender_aware.json \
    --vocab telugu_vocab_verified.json \
    --count 100 \
    --type both
```

The generator can randomly flip the initial ordering of the two names. Thus, both initial orders can occur while preserving the intended grammatical relationship in the subsequent sentence.

Generated files use names of the following form:

```text
telugu_corpus_<count>_<type>_<timestamp>.json
```

For example:

```text
telugu_corpus_50_both_20261007_1026.json
```

When `both` is selected, each item contains the matched canonical and swapped sentences together with metadata such as the selected names, their genders, and the initial ordering.

### Telugu Perplexity Evaluation

`evaluate_telugu_perplexity.py` performs real language-model perplexity evaluation on the generated Telugu corpus.

The evaluation includes:

- Sentence-level perplexity
- Corpus-level, token-weighted perplexity
- Mean and standard deviation of sentence perplexity
- Pairwise perplexity differences
- Pairwise loss differences
- Pairwise comparisons between swapped and canonical sentences

The current evaluation setup includes:

- `bigscience/bloom-560m`
- `allenai/OLMo-1B-hf`

Example:

```bash
python evaluate_telugu_perplexity.py \
    --corpus telugu_corpus_50_both_20261007_1026.json \
    --output telugu_perplexity_results_20261007_1026.json
```

### Current Telugu Evaluation

A preliminary 50-pair Telugu corpus has been evaluated with both models.

#### BLOOM-560M

| Metric | Swapped | Canonical |
|---|---:|---:|
| Corpus PPL | 905.4222 | 832.7482 |
| Mean sentence PPL | 1145.7679 | 1043.1251 |
| Sentence PPL Std. Dev. | 820.1654 | 711.1412 |

Additional pairwise results:

| Metric | Result |
|---|---:|
| PPL difference (Swapped − Canonical) | +72.6740 |
| PPL ratio (Swapped / Canonical) | 1.0873 |
| Mean pairwise PPL difference | +102.6427 |
| Mean pairwise loss difference | +0.085564 |
| Swapped > Canonical | 31 / 50 |
| Canonical > Swapped | 11 / 50 |
| Ties | 8 / 50 |

#### OLMo-1B

| Metric | Swapped | Canonical |
|---|---:|---:|
| Corpus PPL | 3.1983 | 3.1916 |
| Mean sentence PPL | 3.2374 | 3.2316 |
| Sentence PPL Std. Dev. | 0.3133 | 0.3137 |

Additional pairwise results:

| Metric | Result |
|---|---:|
| PPL difference (Swapped − Canonical) | +0.0067 |
| PPL ratio (Swapped / Canonical) | 1.0021 |
| Mean pairwise PPL difference | +0.0058 |
| Mean pairwise loss difference | +0.001819 |
| Swapped > Canonical | 26 / 50 |
| Canonical > Swapped | 16 / 50 |
| Ties | 8 / 50 |

### Telugu Result Summary

**BLOOM-560M** shows an encouraging IOI signal in the preliminary Telugu evaluation: the swapped corpus has 8.73% higher perplexity than the canonical corpus, and the swapped sentence has higher perplexity in 31 of 50 pairs.

**OLMo-1B** shows only a negligible difference: the swapped corpus has 0.21% higher perplexity, with swapped sentences winning in 26 of 50 pairs.

Absolute perplexity values should not be compared across models. The relevant comparison is between swapped and canonical sentences **within the same model**.

These results are preliminary. The Telugu corpus should undergo linguistic inspection before the results are treated as a final validation of the IOI effect. In particular, Telugu case suffixes and other name-specific morphology need to be checked for forms such as `లక్ష్మికి` versus `లక్ష్మికు`.

## Generated Data

Generated corpora are kept inside their respective language directories.

Hindi files may include:

```text
hindi_corpus_*.json
hindi_perplexity_results_*.json
```

Telugu files may include:

```text
telugu_corpus_*.json
telugu_perplexity_results_*.json
```

The timestamped filenames distinguish different generation/evaluation runs.

## Current Status

| Component | Status |
|---|---|
| Hindi vocabulary | Complete |
| Hindi vocabulary candidate generation | Complete |
| Hindi templates | Complete |
| Hindi IOI data generation | Complete |
| Hindi perplexity evaluation | Complete |
| Hindi generated corpora | Available |
| Hindi perplexity results | Available |
| Telugu vocabulary verification | Complete |
| Telugu gender-aware templates | Complete |
| Telugu IOI data generation | Complete |
| Telugu perplexity evaluation | Complete |
| Telugu generated corpora | Available |
| Telugu perplexity results | Available |
| Telugu linguistic quality review | In progress |

## Notes

- The root-level `evaluate_perplexity.py` is intentionally not documented as part of the current language-specific pipeline.
- Hindi has a complete generation and evaluation pipeline.
- Telugu now has a complete computational pipeline, but its generated corpus is still undergoing linguistic quality review.
- Perplexity results are model-relative: swapped and canonical forms should be compared within the same model.
