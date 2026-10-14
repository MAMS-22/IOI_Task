# English IOI circuit discovery for BLOOM

Use `discover_bloom_ioi.py` for the consolidated implementation. The earlier
scripts are useful exploratory work, but do not establish a complete IOI circuit.
Existing results and scripts have been preserved.

This implements the **path-patching branch** of Zhang et al., *The Same but
Different* (ICLR 2025), §2.1, §3 and Appendix B.1, following the original IOI
experiments of Wang et al. It uses the current TransformerLens `TransformerBridge`
API and the original Easy-Transformer English templates. It does not implement
Ferrando & Voita's information-flow-route extraction (§2.2); ordinary attention
probabilities are not that method, and the paper's flow threshold `0.03` must not
be applied to these patching scores.

## Run

From the repository root, with the existing environment:

```bash
PYTHON=/DATA/cs25mtech11017/miniconda3/envs/ioi-ppl/bin/python

# Baseline, original HF logit parity, routing and random-token probes.
$PYTHON English_Task/discover_bloom_ioi.py \
  --offline --verify-hf --stages baseline,functions \
  --output-dir English_Task/english_results/english_baseline

# Complete research run. Defaults: 200 baseline, 50 discovery and 200
# held-out examples PER pattern; all 24 layers and 16 heads.
$PYTHON English_Task/discover_bloom_ioi.py --offline \
  --output-dir English_Task/english_results/english_circuit

# Small end-to-end check: discovery uses just the first four templates.
# It is implementation validation, not a reliable final circuit estimate.
$PYTHON English_Task/discover_bloom_ioi.py --offline \
  --n-per-pattern 30 --discovery-n 4 --validation-n 15 \
  --batch-size 8 --head-roles END,S2 --top-k 2 \
  --mean-donors 4 --random-trials 2 --probe-samples 8 --probe-length 16 \
  --output-dir English_Task/english_results/circuit_smoke

$PYTHON -m unittest discover -s English_Task/tests -v

# Optional real-model intervention and attribution checks (cached weights).
$PYTHON English_Task/tests/verify_bloom_hooks.py
```

Use another output directory for different configurations. `--resume` skips
finished stages only when the saved configuration matches. It preserves stages
saved in `results.json`; a partially finished scan restarts at the start of that
stage. For a faster initial study, omit `residual,heads` and run
`--stages baseline,direct,paths,functions,validate`. Exhaustive scans require
many model forward passes; default runs can take substantial time.

The default device is CUDA when available and otherwise CPU. Float32 preserves
the numerical behavior checked in this environment. `--offline` uses locally
cached weights; omit it to allow model downloads. Dependencies validated here
are in `requirements-circuit.txt`. The vendored Easy-Transformer is a reference;
it is not imported or installed by the new pipeline.

## Data and metric

Each input ends immediately before the answer, e.g.:

```text
ABBA: Then, Mary and John went to the store. John gave a book to
BABA: Then, John and Mary went to the store. John gave a book to
IO = Mary; S = John; score = logit(" Mary") - logit(" John")
```

`A` denotes IO and `B` denotes S in the source templates. ABBA/BABA refer to the
completed sentence including the held-out final A, although the answer is never
fed to the model. The two patterns share lexical choices. Templates are balanced
to within one example. Names must be single tokens **in their actual context**;
the prefix-plus-answer encoding must equal prefix IDs plus the answer ID.

Positions are `IO`, first subject `S1`, its following token `S1+1`, repeated
subject `S2`, and the final prefix token `END`. An explicit BOS is added by
default, consistently across evaluation and interventions; use `--no-bos` to
study sensitivity. Batches group identical lengths and use no padding. The raw
TransformerBridge retains the model's learned LayerNorm weights and original
HF logits; compatibility-mode weight transformations are not required.

Default `--corruption abc` replaces the three name mentions with three distinct
new names, all different from the clean targets. This provides a no-repetition
control with exactly aligned tokens. `--corruption subject_swap` changes only
S2 to IO, as the older scripts do; it reverses the answer roles. Every corrupted
or patched run is scored on the **clean** IO/S token pair. Corrupt accuracy is
therefore a reference measurement, not accuracy against a new task label.

The 15 templates exactly match the short `BABA_TEMPLATES` bank in the vendored
Easy-Transformer source (checked by a regression test). Zhang et al. state that
they use 15 templates, but do not supply the complete lexical pools, sampling
conventions or corruption distribution in the provided paper. This is a method
implementation, not a claim of exact dataset reproduction. Do not hardcode the
paper's head list as a discovered circuit.

## What the stages measure

| Stage | Intervention or measurement | Output |
|---|---|---|
| `baseline` | Unmodified clean/corrupt next-token prediction | Accuracy, zero-rank rate, signed logit difference |
| `residual` | Restore clean residual at each layer and semantic position on corrupt inputs | Denoising localization |
| `heads` | Restore one clean head's Z at each layer/head/selected position | Denoising head effects; all layers by default |
| `direct` | Replace one END head with its corrupt Z while all other attention outputs stay clean | Signed head-to-logit paths; retains both signs |
| `paths` | Replace a sender, freeze intermediate attention outputs, record receiver Q/K/V, replay only receiver inputs on a fresh clean run | Backward candidate tracing and individually confirmed edges |
| `functions` | IOI routing, final-LN-aware logit attribution, repeated vs unique random-token controls | Evidence about copying, suppression, duplication, induction and previous-token behavior |
| `validate` | Mean-ABC ablation on separate examples | Candidate-only, candidate-removed, leave-one-out and matched random sets |

Path tracing first screens END senders into output-head Q at END. It then screens
S2 senders into the selected heads' V at S2, and S1+1 senders into their upstream
induction candidates' K at S1+1. This follows the IOI hypothesis: earlier duplicate/induction
information feeds END query modification through the S2 value route. Screens
use a **group** of receivers in layers later than the sender, then selected sender-to-receiver edges are tested
individually. It is a configurable heuristic search, not an exhaustive search
over all possible graph edges. The default `--routing-threshold 0.1` guides
selection with measured IOI attention: output candidates attend to IO and have
nontrivial vocabulary projection, query-path candidates attend to S2, value-path
candidates attend from S2 to S1 or S1+1, and key-path candidates attend from S1+1
to S1. Only induction-routing candidates are receivers for the previous-token
key-path screen. These are explicit IOI-hypothesis screens, not proof of function.
Set the threshold to zero to perform an unguided causal search. The threshold is
unrelated to the paper's information-flow-route threshold.

`--top-k` (default 8 per tier and per sign for outputs) and `--min-effect` determine its size;
try several settings and discovery seeds, and expand the sender/receiver scans
through `Experiment.scan` if unexplained validation losses remain.

The path implementation recomputes LayerNorm exactly and freezes other heads'
outputs in Z space, which is equivalent to freezing their W_O-projected residual
contributions. It uses no approximate fixed-scale Q/K/V update. By default MLPs
remain free, matching Easy-Transformer's `freeze_mlps=False`; such paths may
include MLP mediation. `--freeze-mlps` restricts analysis to the residual route
with MLP outputs fixed to the clean cache. State that choice when reporting
results. Patching the full residual without freezing other heads is activation
patching, not isolated path patching.

Effects include paired standard errors and separate ABBA/BABA summaries. Ranking
uses **raw** mean changes; normalization by the clean-corrupt mean gap is shown
for comparison only and is undefined for gaps below `1e-6`. Noising effects are
`patched - clean`; denoising effects are `patched - corrupt`. A negative noising
effect indicates a helpful contributor; a positive effect can identify a negative
name mover. Normalized effects can lie outside [-1,1] and must not be clipped.

## Functional evidence and circuit validation

The name-mover test needs attention to IO **and** positive projection toward the
IO logit, supported by a causal output path. A negative name mover should have
negative vocabulary projection; an ablation improvement alone is insufficient.
The S-inhibition hypothesis needs a causal path into name-mover Q and a drop in
the mover's IO-versus-S1 attention preference after corruption. Individual edge
records include that attention change.

For random-token probes, the input is `BOS x1 ... xn x1 ... xn` with unique
random tokens in each half, compared against a non-repeating sequence. The tests
score second-half attention to the same token in the first half (duplication),
the token after that first occurrence (induction), and the preceding token
(previous-token behavior). The last repeated position is excluded from induction
scoring because its successor is outside the first half. Raw scores for both
controls are saved for all heads; they are not thresholded into proven labels.

Logit attribution centers each head's residual output, divides by the observed
final LayerNorm scale, applies the retained learned gamma and reads out with
`W_U[:, IO] - W_U[:, S]`. It measures a clean-run readout contribution, not the
counterfactual effect of removing a head. These two measurements can disagree
because of compensation and downstream nonlinear computations.

Validation uses seed `seed+1` and explicitly excludes baseline/discovery prompts.
Mean-ablation donors use independent RNG `seed+2` and multiple disjoint ABC name
triples per held-out context. Non-name tokens and semantic positions stay fixed.
The candidate-only run retains the chosen head-position nodes and substitutes
donor means everywhere else. The candidate-removed run tests joint necessity.
Leave-one-out drops measure marginal necessity **inside the candidate circuit**.
Random node sets match the candidate's layer and semantic-position counts.

This validates an **attention-node circuit conditional on retained embeddings
and MLPs**. It does not prove completeness/minimality of a graph containing MLPs,
or guarantee that the extracted candidates explain the full IOI mechanism.
Inspect held-out fidelity, random controls, individual edge signs, probe scores,
and both pattern summaries before assigning functional labels. Failure to retain
performance is evidence to expand the circuit, not evidence to relabel the
current candidate set as complete.

## Artifacts

Each run writes `results.json`, containing configuration, package versions,
clean/corrupt token IDs, semantic positions, dataset splits, stage measurements,
candidate nodes, individual edge tests and validation. `candidate_circuit.dot`
exports the measured candidate graph. Render it with Graphviz if installed:

```bash
dot -Tsvg English_Task/english_results/english_circuit/candidate_circuit.dot \
  -o English_Task/english_results/english_circuit/candidate_circuit.svg
```

`plot_bloom_ioi.py` uses Matplotlib and NumPy to export static PNG/PDF path-effect
heatmaps and held-out validation bars. Install `matplotlib` in your plotting
environment if needed (the model runner does not require it):

```bash
$PYTHON English_Task/plot_bloom_ioi.py \
  English_Task/english_results/english_circuit/results.json

$PYTHON English_Task/summarize_bloom_ioi.py \
  English_Task/english_results/english_circuit/results.json
```

To validate a manually refined node set, supply a JSON array of
`{"layer": 18, "head": 7, "role": "END"}` objects, or a previous results file:

```bash
$PYTHON English_Task/discover_bloom_ioi.py --offline --stages validate \
  --circuit English_Task/english_results/english_circuit/results.json \
  --output-dir English_Task/english_results/refined_validation
```

Use a fresh validation seed for a final assessment after tuning on earlier
held-out results. `--head-roles ALL` studies whole-head denoising instead of
individual semantic positions; `ALL` nodes are also supported in supplied circuits.

## Audit of the existing work

| Existing script | Assessment |
|---|---|
| `english_ioi_bloom_baseline.py` | Provisional template bank, including one template with three prefix subject mentions. Use the consolidated data bank for circuit work. |
| `evaluate_abba_baba.py` | Good next-token baseline using the original 15 short templates and BOS. It includes the answer in the input and scores its preceding token; causality makes that score valid, but answer-free inputs simplify all subsequent interventions. |
| `activation_patch_ioi.py` | Correct denoising direction and semantic alignment; only 13 templates, small fixed sample and subject-swap corruption. A localization experiment, not path extraction. |
| `head_activation_patch_ioi.py` | Useful late output-head screen; only layers 13–23 at END. This misses previous-token and duplicate heads in early layers and induction heads acting at S2. |
| `ablate_ioi_heads.py` | Zero ablation is a valid intervention, also used in the paper. Six manually chosen heads tested separately do not establish circuit sufficiency or joint minimality. Positive ablation deltas should prompt investigation of negative contributors. |
| `analyze_ioi_attention.py` | Routing evidence helps interpretation; attention alone does not prove causal function or constitute information-flow-route extraction. |

Perplexity evaluation of completed sentences under `english_data/` is a useful
language/data check. Circuit discovery should use the IOI next-token contrast
metric above, because whole-sentence likelihood mixes the task with general
fluency and includes the expected answer.

## Latest attempt: conditional fidelity and vocabulary predictions

Start with the [second-attempt meeting addendum](english_results/second_attempt/MEETING_ADDENDUM.md)
and [updated figure PDF](english_results/second_attempt/second_attempt_figures.pdf).
The new `refine_bloom_ioi.py` runner prunes whole heads with symmetric per-prompt
margin error, vocabulary agreement and functional-group knockout constraints.
It uses a new development cohort and a fresh 1,000-prompt test, preserving all
earlier results.

The vocabulary-constrained mask retains 137 of BLOOM's 384 attention heads:
100% IO-vs-S accuracy, 93.7% vocabulary top-1 (full model: 96.6%), and 98.5%
of the full mean margin. It retains 35/38 published head IDs, including all
three negative heads. A task-focused variant retains 130 heads with 93.0%
top-1 and 98.3% margin retention. These masks retain each chosen head at all
token positions; they do not reproduce the paper's sparse 38-head graph.

On ten identical whole-head knockouts drawn from the three inventories'
shared heads, the worst pooled gap is 2.187 logits for the previous 33-head
inventory, 0.331 for the task variant, and 0.199 for the vocabulary variant.
This controls the test prompts, knockout subsets and retention granularity.
Finite agreement does not prove completeness. The 95% development agreement
target also falls to 94.3% on the fresh test; this is disclosed without retuning.

The addendum includes exact protocols, source snapshots, matched random controls,
bootstrap intervals and reproduction commands. Use its figures first for the
meeting, and the original functional-probe figures for the mechanistic evidence.

## First completed reproduction for the meeting

The newer checkpointed workflow is `reproduce_bloom_ioi.py`. Start with the
[meeting brief](english_results/meeting_reproduction/MEETING_BRIEF.md),
[seven-page figure PDF](english_results/meeting_reproduction/meeting_figures.pdf),
and [full reproduction report](english_results/meeting_reproduction/REPRODUCTION_REPORT.md).
These supersede the earlier fixed top-k run for reporting current progress;
the older results remain intact for comparison.

The new workflow uses the original vendored lexical pools filtered to single
BLOOM tokens, 400 discovery prompts, 200 independent confirmation prompts,
1,000 sealed test prompts, and 3,000 independent template-conditioned ABC
reference prompts. It screens ABBA and BABA separately, has no head quota,
tests positive/negative/subject-focused Q receivers separately, and includes
subject K/V and induction K paths. Published head IDs are evaluation-only.

The broad search contains all 38 published head-position nodes among 443
candidates. Development-only greedy pruning produces a different 33-node
mask containing 22 published nodes. On the shared final test, its IO-vs-S
accuracy is 99.1% and its retained mean margin is 89.0%, compared with
85.4% and 29.0% for the earlier 33-node mask. These are attention circuits
conditional on retained embeddings and MLPs.

The compact mask's vocabulary top-1 accuracy is 9.9%, versus 96.6% for the
full model; the broad mask retains 81.1%. Matched knockout tests expose a
3.020-logit completeness gap. The compact mask removes all three published
negative heads, so increased intact margin is not evidence of reproducing
the exact functional graph. Report these limitations with the improvement.

Run phases in the order documented in the full report. Use a **new output
directory and a new final test** for further circuit tuning. Each phase saves
configuration, code hashes, raw effects and dependency versions. Regenerate
the meeting artifacts from completed checkpoints with:

```bash
$PYTHON English_Task/report_bloom_reproduction.py
```

The figure generator checks the frozen mask hash, disjoint task splits and
paired evaluation row alignment. Its `compact_circuit.json` and CSV export
contain a measured subset of receiver-input paths, not a proven complete graph.

## References

The measured comparison with the paper's 38-head Figure 2 inventory is in
[`paper_comparison.md`](english_results/english_circuit_seed42/paper_comparison.md).
The reference is an oracle diagnostic with explicitly inferred retention
positions. It is not a blind discovery result. Re-run the common-control test:

```bash
$PYTHON English_Task/compare_paper_circuit.py \
  English_Task/english_results/english_circuit_seed42/results.json
```

- [Zhang et al., paper](https://arxiv.org/abs/2410.09223); supplied PDF is under `paper/`.
- [Wang et al., original IOI study](https://arxiv.org/abs/2211.00593).
- [Easy-Transformer reference code](https://github.com/redwoodresearch/Easy-Transformer), especially `ioi_utils.path_patching`, `ioi_dataset` and `ioi_circuit_extraction`.
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and its [IOI activation-patching demonstration](https://transformerlensorg.github.io/TransformerLens/generated/demos/Main_Demo.html).
