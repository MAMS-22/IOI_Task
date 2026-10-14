# Second attempt: preserve model behavior under circuit reduction

On a new **1,000-prompt test (500 paired contexts)**, the frozen **137-whole-head** circuit achieves **100.0% IO-vs-subject accuracy**, **93.7% vocabulary top-1**, and **98.5% retained mean margin**. Top-token agreement with the full model is **94.3%**; per-prompt margin MAE is **0.524 logits**.

The parallel **conditional task** variant retains **130 whole heads**, with **100.0% IO-vs-S accuracy**, **93.0% vocabulary top-1**, and **98.3% retained margin**. It uses the same conditional margin checks but does not constrain vocabulary predictions. Both masks were frozen before evaluating this shared fresh test.

This is a different and larger circuit representation: a whole head is retained at every token position. It improves the behavior-preservation objective; it does not establish that the paper's sparse 38-head functional circuit has been exactly reproduced. Compare retention units explicitly.

## Common fresh-test comparison

| Condition | Retention | IO > S | Vocabulary top-1 | Margin/full | Top-token agreement | Margin MAE |
|---|---|---:|---:|---:|---:|---:|
| Full model | all 384 heads | 100.0% | 96.6% | 100.0% | 100.0% | 0.000 |
| Previous compact: 33 positions | 33 head-positions | 99.0% | 6.9% | 86.8% | 7.5% | 1.631 |
| Previous compact: 33 whole heads | 33 whole heads | 99.2% | 11.6% | 92.4% | 12.2% | 1.523 |
| Paper 38: inferred positions | 38 head-positions | 99.4% | 3.4% | 81.5% | 4.0% | 1.479 |
| Paper 38: whole heads | 38 whole heads | 99.5% | 3.4% | 81.7% | 4.1% | 1.486 |
| Conditional task circuit | 130 whole heads | 100.0% | 93.0% | 98.3% | 93.2% | 0.576 |
| Vocabulary-preserving circuit | 137 whole heads | 100.0% | 93.7% | 98.5% | 94.3% | 0.524 |

All rows use the same new test prompts and the same independent 3,000-prompt ABC mean bank. Embeddings and MLPs remain present. The published-reference masks are our evaluations with inferred positional conventions, not faithfulness numbers reported by the authors.

The previous 33-head inventory is evaluated both at its original semantic positions and at all positions. This control separates changing the retention granularity from changing which heads are selected; comparison only against the positional circuit would mix those effects.

## What changed, and why

The first pruning pass optimized a lower bound on mean IO−S margin and binary accuracy. That could discard negative contributions and preserve the two-name answer while changing vocabulary predictions. This attempt penalizes both increases and decreases in margin, measures per-prompt error and top-token agreement, and checks conditional behavior under six functional-group knockouts.

Both searches began with all 384 whole heads. Eighteen anchor heads were selected from prior independently inferred functional families (up to three per family), using routing, copy-readout sign and repeated-token probes. Those anchors were protected and defined the development knockouts; published IDs were not used. Block removal (16/4/1 heads) used the same 200 new development prompts. The vocabulary constraint caused limited early pruning on development, so the task-only variant was added before any fresh test evaluation. Both circuits were frozen before that test; its results did not revise either mask.

Per ordering, the development constraints were: intact absolute mean bias <=5% and mean absolute error <=10% of the full-model mean margin; >=95% top-token agreement; no more than one percentage point IO-vs-S accuracy loss or two points vocabulary top-1 loss. Under each anchor knockout, absolute mean bias <=8% and MAE <=15%. These are empirical tolerances, not a proof of minimality or completeness.

The development targets do not all generalize to the fresh test. In particular, the vocabulary variant's pooled top-token agreement is 94.3%, below its 95% development requirement, and its vocabulary top-1 loss is 2.9 percentage points. Per-ordering checks are recorded in `metrics.json`. These deviations are reported without retuning either mask on the final test.

## Completeness evidence

The largest pooled absolute gap for the six development-selected knockout groups on fresh test prompts is **0.181 logits** (subject_focused_END_candidate). For **ten previously unused random knockout subsets**, it is **0.374 logits** (unseen_random_8). The largest per-ordering gap across these tests is **0.383 logits** (unseen_random_8, BABA).

For the conditional task variant, the largest pooled gap is **0.237** for anchor groups and **0.592 logits** for the same ten random subsets. These random subsets were sampled from heads retained by both circuits, so the two new variants receive exactly the same knockouts. They do not cover subsets unique to either circuit.

The task variant's largest per-ordering mean gap is **0.704 logits** (unseen_random_8, BABA). Worst per-prompt MAE over the ten unseen random subsets is **0.630 logits** for the vocabulary variant and **0.717** for the task variant. Mean gaps alone can hide per-prompt errors and ordering cancellation.

A separate controlled comparison uses **ten identical whole-head knockout subsets shared by all three inventories** on these same fresh prompts. Worst pooled absolute gaps: previous 33 heads retained at all positions **2.187 logits**; conditional task variant **0.331**; vocabulary variant **0.199**. This comparison controls prompts, knockout subsets and ablation granularity. It covers only subsets of heads shared by the three inventories, so it is a limited completeness diagnostic.

Finite conditional agreement is evidence of robustness, not exhaustive completeness. The first attempt's 3.020-logit worst gap used a different cohort, positional mask and knockout subsets; do not treat a difference between those maxima as a controlled improvement estimate. The common fresh-test intact comparison above is controlled.

## Comparison with the paper

The vocabulary-preserving inventory overlaps **35/38** published heads; the conditional task inventory overlaps **35/38**. This is head-identity overlap only: a whole-head retention mask does not prove the published positional role or exact connecting paths. The prior discovery run's functional probes and receiver-input experiments remain the evidence for those mechanisms.

| Published family | Vocabulary / published | Task / published | Missing from vocabulary variant | Missing from task variant |
|---|---:|---:|---|---|
| duplicate | 1/3 | 1/3 | 0.9, 7.8 | 0.9, 7.8 |
| previous | 5/5 | 5/5 | none | none |
| induction | 9/10 | 9/10 | 10.11 | 10.11 |
| inhibition | 6/6 | 6/6 | none | none |
| name_mover | 11/11 | 11/11 | none | none |
| negative | 3/3 | 3/3 | none | none |

## Meeting statement

“The first compact circuit preserved the binary IOI decision but lost vocabulary behavior. I changed the pruning objective to symmetric per-example fidelity and added conditional knockout checks. I then evaluated a frozen whole-head circuit on a completely new test set. This provides a stronger behavioral reproduction, with an explicit cost in circuit size and a remaining gap to the paper's exact sparse functional graph.”

Show [second_attempt_figures.pdf](second_attempt_figures.pdf) first, then use the original [functional-probe figures](../meeting_reproduction/meeting_figures.pdf) for mechanistic evidence. The exact masks, anchors, every pruning decision, fresh splits, source snapshots and final measurements are saved alongside this brief.

The next step is to identify which additional heads support general completion versus IOI routing, and reduce head-position retention under these symmetric and conditional checks. The exact author corpus, exact positional mask and sparse functional graph remain unresolved; this result does not replace those reproduction targets.

## Reproduction

```bash
python English_Task/refine_bloom_ioi.py --phase prune --device cuda:0
python English_Task/refine_bloom_ioi.py --phase prune --objective task --output-dir English_Task/english_results/second_attempt_task --device cuda:1
python English_Task/refine_bloom_ioi.py --phase evaluate --device cuda:0
python English_Task/report_second_attempt.py
```

Completed checkpoints are protected from overwrite. For another attempt, use a new output directory and `--seed`, and pass `--exclude-run English_Task/english_results/second_attempt` to exclude this round's prompts as well as the first round's. Do not tune on either final test. The report checks source snapshots, frozen-mask hashes, disjoint prompts, paired row alignment and the initial all-head identity under intact and knockout conditions. The earlier six real-model hook checks and 14 CPU regression tests still apply to the unchanged intervention helpers. Confidence intervals resample matched contexts within fixed templates; they do not include variation from discovery seeds or unseen template families.
