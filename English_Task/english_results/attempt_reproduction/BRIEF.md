# English BLOOM IOI reproduction: meeting brief

The validated pipeline now discovers and prunes an English IOI attention circuit independently of the paper's head IDs. On **1,000 fresh prompts (500 matched contexts)**, the frozen 33-node circuit achieves **99.1% IO-vs-subject accuracy** and retains **89.0% of the full model's mean logit difference** (context-cluster bootstrap 95% interval 86.7%–91.3%). Its vocabulary top-1 accuracy is **9.9%**. Report all three metrics together.

This is a method and functional reproduction for BLOOM-560m in English. It is not an exact reproduction of the authors' corpus, all languages, information-flow results, or a proof of completeness/minimality.

## Show these three things

1. Open [meeting_figures.pdf](meeting_figures.pdf), pages 1–2: a controlled comparison on the same untouched test prompts and independent ABC ablation means. Page 7 displays a measured subset of receiver-input paths in the compact mask.
2. Show page 3: the broad search recovered 38/38 paper-position nodes within 443 candidates, including 20/20 final-position heads within 58 direct candidates. The compact mask retains 22/38. Paper IDs were loaded only for evaluation, not used to choose heads. Coverage alone is not a sparse graph reproduction.
3. Show pages 4–5: independent copying/duplicate/induction probes and token/order interventions. For example, patching the subject-focused heads from the name-swapped input changes the mean original-target margin from 5.35 to -0.84; changing order gives 0.67. These are causal group interventions, not a claim that each head has a fully isolated role.

## Controlled final-test results

| Condition | Retained nodes | IO > S | Vocabulary top-1 | Mean IO−S | Margin/full |
|---|---:|---:|---:|---:|---:|
| Full model | all | 100.0% | 96.6% | 5.457 | 100.0% |
| Previous 33 nodes | 33 | 85.4% | 4.1% | 1.581 | 29.0% |
| Paper 38 (reference) | 38 | 99.8% | 4.5% | 4.534 | 83.1% |
| Confirmed broad | 443 | 100.0% | 81.1% | 4.767 | 87.4% |
| Functional core | 125 | 99.7% | 58.3% | 4.318 | 79.1% |
| Pruned discovery | 33 | 99.1% | 9.9% | 4.859 | 89.0% |

All circuit conditions retain embeddings and MLPs. A node is a head at a semantic token position; unique heads and nodes are different counts. The pruned mask contains 33 unique heads. The paper's 38-head mask here uses inferred positional retention conventions, not an author-supplied ablation mask. The paper reports full-model English IO-vs-S accuracy of 100%/100% and top-1 of 96%/91% for normal/flipped prompts; our ABBA/BABA labels are not verified to map to those labels.

Pruning optimized IO-vs-S accuracy and mean margin, not vocabulary top-1 or completeness. It removes all three published negative heads. Its stronger intact margin than the published-reference mask does not establish a more accurate functional graph.

## Three-minute explanation

“The previous implementation found some of the expected heads but discarded weak paths with a fixed top-k rule and pooled the two prompt orderings. I replaced that with separate paired screens for ABBA and BABA, independent confirmation, separate positive/negative/subject receiver groups, and query/key/value path tests. I also replaced evaluation-derived ablation donors with 3,000 independent, template-conditioned ABC reference prompts.

“I first found a broad causal mask, then greedily pruned it using only 200 development prompts. I froze that mask before evaluating 1,000 new prompts. The old circuit, paper reference, new circuit, functional core and five matched random masks are all tested under identical controls. The headline is 99.1% name-choice accuracy and 89.0% retained margin, versus 29.0% for the old mask. The random-mask margin range is -32.2%–1.8%.

“We also reproduce several functional behaviors and causal sensitivity to the repeated name and its order. I would call this a stronger English method reproduction, with remaining gaps in the exact functional graph, author-data equivalence and circuit completeness.”

## Questions to expect

- **Is the circuit complete?** No. It fails the matched C\K versus M\K stress tests: the largest observed absolute mean gap is 3.020 logits (random_subset_3), showing substantial unmodeled compensation. Page 6 shows the measured gaps. Strong intact name-choice performance does not imply a complete causal graph.
- **Did you just use the published heads?** No. The selection code uses effects, routing and random probes. The published mask is a clearly labeled benchmark. Pruning may remove a published head when other routes compensate.
- **Why more than 38 heads?** The causal search admits generic support routes, while the paper's circuit groups heads by function. Greedy pruning improves sparsity but does not establish the smallest sufficient mask. The functional core is a separate predefined sensitivity analysis.
- **Does 100% name-choice accuracy mean the same output as the model?** No. IO-vs-S checks only two names. Vocabulary top-1 and retained margin are reported separately.
- **What changed besides the algorithm?** The lexical pools now follow the vendored Wang dataset and are filtered to single BLOOM tokens: 90 names, eight places and seven objects, with 15 templates per ordering. Because the corpus differs, all old/new/reference masks were re-evaluated on one shared fresh split.
- **Is this the paper's information flow?** No. Attention probabilities, causal patching and OV capability tests are reported. Ferrando/Voita flow and cross-language comparison remain unimplemented.

## Next experiments after the meeting

Obtain the authors' exact English corpus and run protocol; isolate remaining role assignments with controlled receiver-input interventions; examine the largest matched-knockout discrepancies; repeat discovery across seeds and templates; then reproduce the information-flow calculation before adding languages. Preserve this final test as a sealed benchmark: later changes need a new test split.

See [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md) for methods, uncertainty, commands and raw checkpoints.
