# English BLOOM IOI reproduction report

Generated 2026-10-14T03:20:25.803004+00:00. Model: `bigscience/bloom-560m`. No weights were trained or fine-tuned.

## Final test

| Condition | Retained nodes | IO > S | Vocabulary top-1 | Mean IO−S | Margin/full |
|---|---:|---:|---:|---:|---:|
| Full model | all | 100.0% | 96.6% | 5.457 | 100.0% |
| Previous 33 nodes | 33 | 85.4% | 4.1% | 1.581 | 29.0% |
| Paper 38 (reference) | 38 | 99.8% | 4.5% | 4.534 | 83.1% |
| Confirmed broad | 443 | 100.0% | 81.1% | 4.767 | 87.4% |
| Functional core | 125 | 99.7% | 58.3% | 4.318 | 79.1% |
| Pruned discovery | 33 | 99.1% | 9.9% | 4.859 | 89.0% |

Accuracy means IO logit > S logit. Top-1 means IO is first across the complete vocabulary. Margin/full is the ratio of mean logit differences; it is not bounded to [0,1]. Empty-adjusted recovery is stored separately in `evaluate_discovered_test.json`. Shown final-test intervals resample the 500 matched ABBA/BABA contexts as clusters, stratified within the 15 fixed templates, preserving both model/circuit and ordering dependence. They do not estimate template-bank or discovery-seed uncertainty. Zero observed errors give a degenerate bootstrap interval and do not prove population perfection. Auxiliary Wilson intervals are stored as prompt-independence heuristics, not used in figures.

| Ordering | Full accuracy | Full top-1 | Pruned accuracy | Pruned top-1 | Pruned margin/full |
|---|---:|---:|---:|---:|---:|
| ABBA | 100.0% | 97.2% | 99.2% | 9.0% | 78.9% |
| BABA | 100.0% | 96.0% | 99.0% | 10.8% | 102.2% |

## Protocol and validation

- Discovery: 400 prompts; confirmation/development: 200; sealed final test: 1,000; ABC mean bank: 3,000 (100 per template and ordering). Task prompts are explicitly disjoint, and previously evaluated prompts were excluded.
- Seed: 20261014; the manifest stores every prompt, token ID, answer token and semantic index. Explicit BOS and float32 are used. The vocabulary comes from the vendored Redwood/Wang dataset and is filtered for leading-space single BLOOM tokens.
- Discovery uses two-sided paired mean/SE normal-approximation tests with BH screening separately for ABBA and BABA. The accepted union avoids ordering cancellation. Direct discovery floor: .02 logits; upstream/confirmation floor: .001; FDR screen: .05. This screen is not proof of functional relevance and does not control a global error rate over all sequential stages.
- Path patching freezes other attention outputs, corrupts a sender, allows intervening MLP/LayerNorm recomputation, records receiver Q/K/V and replays that input into a fresh clean run. The intervention is nonlinear and uses actual BLOOM hooks.
- Six real-model checks passed: actual Q/K/V weight layouts, Z/result equivalence, identical-donor path identities, exact frozen-MLP residual/LN reconstruction, affine/centered readout, and all-attention retention identity. Fourteen CPU regression tests passed.
- `hook_validation.json` records the executed assertions and numerical errors on 2 diagnostic prompts. The largest recorded absolute error is 0.000351, checked with absolute tolerance 0.0005 and relative tolerance 0.0001. These sanity checks do not cover every prompt or every Q/K/V layer.
- Confirmed broad mask: 443 nodes/289 unique heads. Functional core: 125 nodes/118 heads. Pruned: 33 nodes/33 heads. Semantic position retention is explicit. Embeddings and MLPs remain present.
- Pruning uses only development prompts: marginal ranking followed by blocks of 16/4/1, with fixed per-ordering margin and accuracy targets stored in `prune.json`. The budget flag is `False`. Finite greedy pruning does not prove minimality. The mask hash in final evaluation must match `prune.json`.

The development margin targets were ABBA 4.837 (79.9% of full) and BABA 4.176 (90.0% of full). ABBA starts below the nominal 90% target, so its floor is the initial broad-mask margin. The algorithm does not claim to attain 90% for both patterns. Vocabulary top-1 and completeness were evaluated but not optimized.

## Search coverage

| Receiver route | Discovery senders | Independently confirmed | Individual edge tests |
|---|---:|---:|---:|
| positive_output_q | 160 | 150 | 3745 |
| negative_output_q | 140 | 126 | 993 |
| subject_end_q | 191 | 157 | 2190 |
| subject_end_v | 181 | 160 | 2599 |
| subject_end_k | 194 | 175 | 2794 |
| induction_k | 24 | 22 | 688 |

Individual edge effects are stored without treating every tested edge as established. Weak group effects and individual effects need not agree under interaction or cancellation. Functional labels are provisional and may differ from the paper even for a matching head.

| Paper family | Paper nodes | Broad recovered | Core recovered | Pruned recovered |
|---|---:|---:|---:|---:|
| duplicate | 3 | 3 | 3 | 1 |
| previous | 5 | 5 | 5 | 2 |
| induction | 10 | 10 | 10 | 6 |
| inhibition | 6 | 6 | 6 | 6 |
| name_mover | 11 | 11 | 11 | 7 |
| negative | 3 | 3 | 3 | 0 |

Overlap is measured at the paper's inferred semantic positions. The broad screen recovers 38/38 paper nodes, but also many extras. The pruned mask recovers 22/38. Neither head overlap alone nor an ablation score proves the paper's exact graph.

## Functional evidence

`functions.json` stores separate IO and S readout contributions, routing attention, repeated/unique random-token probes and positive/negative OV capability scores over all 90 names. `mechanisms.json` independently repeats functional measurements on 200 confirmation prompts and stores per-example attention/projection scatter values and correlations.

OV probes use the name residual after block 0, each head's own LayerNorm, W_V/W_O without projection biases, and the actual final LN/unembedding. They are capability tests, not direct evidence of causal usage. Random-token controls use BLOOM vocabulary tokens; these are an adapted diagnostic, not an exact replication of the authors' language-specific sampling protocol.

Joint activation replacement in the subject-focused END heads gives these original-target mean margins: original 5.349; name swap -0.844; initial-order swap 0.667; both -5.924. Order changes alter which first name is repeated, while name changes preserve that location. Because BLOOM uses ALiBi, this experiment does not copy a learned position embedding. The intervention is joint END-output replacement, not exclusive isolation of each signal along a Q path.

The supplemental subject-only corruption changes S2 to IO and leaves both initial names fixed. It supports 60 direct END candidates. It does not alter the primary circuit. Its complete effects are in `specificity.json`.

The compact mask retains 22/38 published nodes and removes all three published negative heads. Its higher retained mean margin than the paper-reference mask is therefore not evidence of a more accurate functional graph: removing negative contributions can increase the margin. The broad mask is a coverage result, not a sparse recovery of 38 heads.

## Completeness stress tests

Matched ablations compare C\K with M\K on 1000 test prompts. K includes positional/functional groups and ten fixed random subsets. The largest pooled gap is 3.020 logits, demonstrating that the compact mask is incomplete under these tests. Passing finite tests would not establish exhaustive completeness, and greedy pruning does not establish conditional minimality. Group masks reflect our provisional roles, not a one-to-one validation of paper annotations.

## What is reproduced and what remains

The paper's full-model English accuracy, several head capabilities, multiple causal pathways and token/order sensitivity can be compared. The paper's normal/flipped labels are preserved as published rather than mapped to our ABBA/BABA labels without verification. The authors' exact corpus and positional ablation mask have not been acquired. Their paper does not provide an equivalent BLOOM circuit-only faithfulness score; the GPT-2 87% figure from Wang is not a BLOOM target. All-language circuits, cross-language comparisons and information-flow ratios are outside this English reproduction.

Primary references: [supplied paper](../../../paper/11814_The_Same_but_Different_S.pdf), [author preprint](https://arxiv.org/html/2410.09223v1), [original IOI paper](https://arxiv.org/html/2211.00593v1), [Redwood Easy-Transformer](https://github.com/redwoodresearch/Easy-Transformer), [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens). The final supplied PDF governs the 38-head transcription in `paper_bloom_reference.json`; the preprint may differ.

## Commands and artifacts

Run from the repository root with the `ioi-ppl` environment. The experiment runner sets model downloads offline. Start in a new output directory for a new discovery seed; completed checkpoints are never silently overwritten.

```bash
python English_Task/reproduce_bloom_ioi.py --phase prepare
python English_Task/reproduce_bloom_ioi.py --phase direct --device cuda:0
python English_Task/reproduce_bloom_ioi.py --phase functions --device cuda:1
# Run each of the six --target names above with --phase path.
# induction_k requires completed subject_end_v and subject_end_k.
python English_Task/reproduce_bloom_ioi.py --phase assemble --device cuda:0
python English_Task/reproduce_bloom_ioi.py --phase prune --device cuda:0
python English_Task/reproduce_bloom_ioi.py --phase mechanisms --device cuda:1
python English_Task/reproduce_bloom_ioi.py --phase specificity --device cuda:2
python English_Task/reproduce_bloom_ioi.py --phase evaluate --evaluate-kind discovered --split test --device cuda:0
python English_Task/reproduce_bloom_ioi.py --phase knockouts --split test --device cuda:1
python English_Task/report_bloom_reproduction.py
python -m unittest discover -s English_Task/tests -p 'test_*.py'
python English_Task/tests/verify_bloom_hooks.py --device cuda:0
```

Each checkpoint records source hashes, configuration, timing and dependency versions. Versions: `{"torch": "2.6.0+cu124", "transformers": "5.18.0", "transformer_lens": "4.2.0"}`. `meeting_metrics.json` records input artifact SHA-256 hashes and statistical intervals. `meeting_figures.pdf` contains seven scientific figure pages; PNG counterparts are in `figures/`. `compact_circuit.json` and `compact_circuit_edges.csv` export the mask and the supported subset of measured paths. `MEETING_BRIEF.md` is the short talk track. Preserve these artifacts and the final mask hashes when reporting results.
