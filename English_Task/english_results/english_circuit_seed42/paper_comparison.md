# BLOOM English IOI: comparison with the supplied paper

Reference: Zhang et al., *The Same but Different*, ICLR 2025, Figure 2 (page 5), Figure 6 (page 17), Figure 7 (page 18), Table 1 and Appendix B.1. The head inventory was transcribed from the supplied PDF, not inferred from a web summary. All 38 IDs in Figure 2 are black/shared and therefore apply to English.

## What is already reproduced

The unmodified model prefers IO over S on all 400 baseline prompts. Our full-vocabulary top-1 rate is 91% in each ABBA/BABA pattern; the paper reports 96%/91% for normal/flipped inputs. Its exact sampling, vocabulary and normal/flipped mapping must be established before interpreting that difference.

Our direct path scan has strong effects on heads also visible in Figure 6, including 18.7, 19.4, 21.2, 22.3 and negative contributors 17.9, 19.15, 21.6. The mismatch is substantially in selecting and tracing the circuit, not evidence that BLOOM fails to execute the task.

## Figure 2 inventory versus selected nodes

| Functional class | Recovered / listed | Missing from selected circuit |
|---|---:|---|
| Duplicate token | 1/3 | 0.9, 7.8 |
| Previous token | 4/5 | 4.0 |
| Induction | 5/10 | 6.12, 7.6, 7.10, 8.13, 10.11 |
| S-inhibition | 3/6 | 19.4, 21.2, 22.3 |
| Name mover | 6/11 | 19.8, 20.8, 20.12, 21.9, 21.10 |
| Negative name mover | 3/3 | None |

22 of 38 reference head-position nodes occur in our 33-node circuit. The other 11 selected nodes are outside the Figure 2 inventory; this does not by itself prove they are false positives. Exact retention positions are inferred from Wang/Easy-Transformer conventions, because the BLOOM paper does not provide an executable circuit mask.

## Controlled reference-circuit test

I evaluated the current circuit, the paper inventory at inferred semantic positions, their union, and the reference whole heads on the same 400 prompts with identical ABC donors for every condition. All embeddings and MLPs remain active. This is an oracle diagnostic, not a blind discovery result or a metric reported by Zhang et al.

| Condition | IO > S accuracy | IO top-1 | Mean IO−S logit | Full-model margin retained |
|---|---:|---:|---:|---:|
| full | 100.00% | 91.75% | 5.246 | 100.0% |
| current_nodes | 81.50% | 4.75% | 1.518 | 28.9% |
| paper_role_nodes | 99.25% | 3.75% | 4.106 | 78.3% |
| union_role_nodes | 99.25% | 22.75% | 4.166 | 79.4% |
| paper_whole_heads | 99.25% | 3.50% | 4.144 | 79.0% |

The previous 83.5% current-circuit accuracy used different control draws. The common-control rerun is 81.5%; this illustrates control sensitivity and is not a change in full-model performance. The reference improves pairwise accuracy to 99.25% and margin retention to 78.3%. Retaining the reference whole heads raises margin retention only to 79.0%, providing little support for widening positions alone as the primary fix.

The reference is much more faithful on BABA (91.6% margin retained) than ABBA (67.3%). Its low IO top-1 rate under this ablation protocol shows that pairwise success is not full-vocabulary reproduction.

Important comparison limits: the supplied BLOOM paper does not publish the same circuit-only faithfulness metric, a mean-ablation retention mask, or a complete lexical/sampling specification. The often-cited 87% circuit faithfulness belongs to Wang et al.’s GPT-2-small circuit and is not a numerical BLOOM target.

## Donor-count sensitivity

A second common-control run used 64 rather than eight donors per context. The current circuit reached 84.75% accuracy and retained 30.3% of the full logit margin; the reference reached 99.25% and retained 79.7%; the union reached 99.25% and retained 80.6%. Reference whole-head retention preserved 80.4%. The head-selection conclusion is stable across these two control settings.

The reference top-1 rate remained only 4.0%, versus 3.75% with eight donors; simply increasing donor count does not fix full-vocabulary behavior. The independent template-conditioned ablation protocol and additional non-IOI support need separate investigation. The 64-donor reference retained 93.6% of BABA margin but only 68.2% of ABBA margin, so pattern-specific diagnostics are a priority. Raw data are in `paper_comparison_donors64.json`.

## Specific causes established in our implementation

1. **Top-k pruning excludes clear movers.** 19.8, 20.8 and 20.12 have mean direct noising changes −0.096, −0.128 and −0.153 logits, with positive END IO−S attribution. They pass the routing screen but rank below the eight positive output candidates.

2. **A global raw threshold excludes smaller but measurable routes.** Duplicate heads 0.9 and 7.8 have value-path effects −0.0105 ± 0.0023 SE and −0.0118 ± 0.0017 SE. Both are below our 0.02-logit cutoff despite clean S2→S1 attention 0.407/0.313 and random duplicate-minus-control scores 0.129/0.170. Previous-token head 4.0 has a key-path effect −0.0043 ± 0.00055 SE and strong predecessor attention, and is also excluded.

3. **The pooled effect can hide condition-dependent behavior.** Mover 21.9 has ABBA/BABA changes −0.0394/+0.0232 logits; 21.10 has −0.0931/+0.0106. Ranking only the pooled mean loses such behavior. The receiver screen also combines positive and negative output heads, which can mask individual contributions.

4. **The serial route assumption is too narrow.** 19.4 and 21.2 have strong direct effects (−0.474/−0.466 logits) but low END→IO attention and are excluded as output candidates. They do not survive the query-path top-k stage. Head 22.3 has direct effect −0.276, but cannot be tested as a query sender because no selected receiver is in a later layer. A paper-labelled inhibition head need not influence our selected name-mover queries through this one route.

5. **Only one upstream branch is searched.** We trace S2 senders into inhibition V, but omit inhibition K and other plausible causal routes. Wang et al.’s Appendix C analyzes inhibition keys separately. Functional evidence currently measures means and IO−S projection, not all of the copying and positional-counterfactual experiments used in that work.

6. **Our validation distribution differs from the reference mean-ablation procedure.** We average eight disjoint-name donors within each exact context. Wang uses an independent ABC reference distribution with template-conditioned means. Different means can change the outcome; do not attribute all of the remaining gap to missing heads without this control.

## Recommended order of work

1. Freeze a reproduction protocol: exact author data/code if obtainable; otherwise document the original 15 templates, BLOOM-filtered lexical pools, BOS policy, input prefixes, corruption design, and equal ABBA/BABA counts. Check the model’s baseline and HF parity before any interventions. Do not tune lexical choices merely to hit the published top-1 percentages.

2. Fix discovery selection before increasing runtime: remove fixed per-family quotas; keep candidate effects for each pattern/sign/receiver family; include supported direct contributors even when they do not pass a name-mover attention screen; compare simultaneous and individual effects. Treat routing as evidence for classifying, not a hard exclusion rule.

3. Trace receiver families separately and include both K and V into S-inhibition heads. Explicitly investigate direct output paths of 19.4, 21.2 and 22.3. Test missing weak duplicate/induction/previous routes with larger samples and independent confirmation. Search for compensating heads after group ablations.

4. Match the functional experiments: positive and negative OV copying tests, separate IO and S vocabulary projections versus corresponding attention, previous/duplicate/induction random-token probes, joint inhibition ablations, and token-versus-position counterfactuals. BLOOM uses ALiBi, so do not port GPT-2 learned positional-embedding claims without a BLOOM-specific test.

5. Establish a stable mean-ablation protocol using an independent ABC bank with template-conditioned means and aligned token positions. Report donor-count sensitivity, per-pattern margin retention, pairwise accuracy and full-vocabulary top-1 separately. Increase discovery to at least 200–500 examples per pattern and repeat independent seeds once the route/selection issues are fixed.

6. Freeze the resulting circuit and evaluate on new data. The 400 prompts used here are now development/diagnostic data. Test faithfulness and matched knockout behavior F(C\K) versus F(M\K) for classes, random subsets and greedy subsets; leave-one-out alone does not establish Wang’s completeness/minimality criteria.

7. If reproducing the information-flow panels, implement the actual Ferrando–Voita flow method and its threshold τ=0.03. Raw attention plots are not a substitute. English/Chinese correlations require the Chinese and unrelated-control datasets and are outside the present English-only circuit comparison.

## Artifacts and commands

- `paper_comparison.json`: inventory, recovered/missing nodes, functional measurements and common-control evaluation.
- `paper_reference_nodes.json`: a `--circuit` compatible diagnostic reference mask.
- `../../paper_bloom_reference.json`: transcription and position assumptions.

```bash
python English_Task/compare_paper_circuit.py \
  English_Task/english_results/english_circuit_seed42/results.json
```

Sources: [Zhang et al.](https://arxiv.org/abs/2410.09223), [Wang et al.](https://arxiv.org/html/2211.00593v1), [Easy-Transformer source](https://github.com/redwoodresearch/Easy-Transformer).
