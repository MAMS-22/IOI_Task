# English BLOOM IOI measurements

Model: `bigscience/bloom-560m`. Discovery seed: 42. Discovery: 100 prompts; held-out evaluation: 400 prompts.

This is the English path-patching branch of the supplied paper. The lexical pools and sampling are not verified as the authors’ exact dataset. Candidate nodes are hypotheses, not automatically proven functional heads.

Completed stages: baseline, direct, paths, functions, validate.

## Baseline

| Pattern | Examples | IO > S | IO top-1 | Mean logit difference ± SE |
|---|---:|---:|---:|---:|
| all | 400 | 100.0% | 91.0% | 4.973 ± 0.084 |
| ABBA | 200 | 100.0% | 91.0% | 5.400 ± 0.123 |
| BABA | 200 | 100.0% | 91.0% | 4.546 ± 0.107 |

## Held-out attention-node validation

Embeddings and MLPs remain active in every condition. Outside-circuit attention nodes are replaced by the mean of independent disjoint-ABC donors in the same context.

| Condition | IO > S | IO top-1 | Mean logit difference ± SE |
|---|---:|---:|---:|
| full | 100.0% | 91.8% | 5.246 ± 0.087 |
| empty_attention | 54.2% | 0.0% | 0.199 ± 0.082 |
| candidate_only | 83.5% | 4.8% | 1.537 ± 0.077 |
| candidate_removed | 56.5% | 9.2% | 0.294 ± 0.075 |
| random_0 | 48.8% | 0.0% | -0.040 ± 0.081 |
| random_1 | 63.5% | 0.0% | 0.593 ± 0.078 |
| random_2 | 75.5% | 0.0% | 1.104 ± 0.080 |

| Pattern | Attention-dependent gap recovered | Full-model logit difference retained |
|---|---:|---:|
| all | 26.5% | 29.3% |
| ABBA | 24.2% | 26.8% |
| BABA | 29.4% | 32.3% |

Accuracy alone can conceal lost logit margin. Compare recovered margin against random controls and both name-order patterns. This establishes only attention-node fidelity under this ablation distribution, not full-model completeness.

## Candidate nodes and functional evidence

Duplicate/induction probe columns are repeated-sequence score minus non-repeating control score. Previous-token behavior is the absolute score on the non-repeating control, where these heads should also work. Attribution is measured at END even when a retained node is at another position.

| Head | Retained position | END IO−S attribution | END→IO | END→S2 | Duplicate probe Δ | Induction probe Δ | Previous-token score |
|---|---|---:|---:|---:|---:|---:|---:|
| 5.0 | S1+1 | +0.011 | +0.000 | +0.000 | -0.000 | -0.000 | +0.939 |
| 6.0 | S1+1 | -0.000 | +0.000 | +0.000 | -0.000 | -0.000 | +0.992 |
| 6.2 | S1+1 | +0.023 | +0.000 | +0.000 | +0.000 | -0.000 | +0.990 |
| 7.0 | S1+1 | +0.010 | +0.000 | +0.002 | -0.000 | -0.000 | +0.976 |
| 7.12 | S2 | +0.004 | +0.018 | +0.018 | -0.007 | +0.961 | +0.005 |
| 8.9 | S2 | +0.000 | +0.007 | +0.007 | +0.003 | +0.436 | +0.018 |
| 8.11 | S2 | +0.001 | +0.002 | +0.006 | -0.000 | +0.926 | +0.000 |
| 9.11 | S2 | -0.000 | +0.003 | +0.015 | -0.004 | +0.671 | +0.008 |
| 10.10 | S2 | +0.001 | +0.003 | +0.001 | +0.007 | +0.931 | +0.007 |
| 11.10 | S2 | -0.003 | +0.002 | +0.002 | +0.286 | +0.006 | +0.002 |
| 12.4 | END | -0.010 | +0.014 | +0.382 | +0.000 | +0.001 | +0.025 |
| 12.6 | END | +0.084 | +0.033 | +0.208 | +0.006 | +0.119 | +0.007 |
| 12.10 | S2 | +0.004 | +0.014 | +0.006 | +0.009 | +0.018 | +0.013 |
| 13.13 | S2 | +0.000 | +0.008 | +0.002 | +0.045 | +0.211 | +0.014 |
| 14.6 | END | +0.092 | +0.032 | +0.668 | +0.000 | +0.006 | +0.023 |
| 14.12 | END | -0.006 | +0.030 | +0.150 | -0.009 | +0.017 | +0.030 |
| 16.5 | END | +0.247 | +0.046 | +0.451 | +0.001 | +0.004 | +0.006 |
| 16.10 | END | -0.060 | +0.242 | +0.376 | -0.001 | +0.103 | +0.033 |
| 16.13 | END | +0.390 | +0.351 | +0.018 | +0.019 | +0.159 | +0.010 |
| 16.14 | END | +0.233 | +0.194 | +0.027 | -0.002 | +0.000 | +0.003 |
| 17.8 | END | +0.033 | +0.009 | +0.285 | +0.003 | +0.001 | +0.019 |
| 17.9 | END | -0.187 | +0.813 | +0.019 | -0.002 | +0.213 | +0.003 |
| 18.7 | END | +0.511 | +0.140 | +0.537 | +0.000 | +0.064 | +0.002 |
| 18.8 | END | +0.336 | +0.555 | +0.034 | -0.001 | +0.055 | +0.011 |
| 19.11 | END | +0.561 | +0.444 | +0.029 | -0.004 | +0.834 | +0.016 |
| 19.15 | END | -0.171 | +0.689 | +0.019 | -0.004 | +0.512 | +0.011 |
| 20.9 | END | +0.345 | +0.565 | +0.052 | -0.001 | +0.726 | +0.022 |
| 20.10 | END | +0.446 | +0.586 | +0.044 | -0.001 | +0.671 | +0.024 |
| 20.11 | END | +0.383 | +0.331 | +0.067 | +0.002 | +0.695 | +0.015 |
| 21.6 | END | -0.387 | +0.806 | +0.062 | +0.002 | +0.566 | +0.000 |
| 21.8 | END | -0.140 | +0.198 | +0.056 | -0.001 | +0.002 | +0.008 |
| 22.10 | END | -0.049 | +0.239 | +0.085 | +0.001 | +0.007 | +0.002 |
| 22.12 | END | -0.142 | +0.173 | +0.045 | -0.000 | +0.003 | +0.007 |

179 individual sender/receiver paths measured. Signs, paired standard errors and receiver attention changes are in `results.json`. The graph is in `candidate_circuit.dot`; path and validation figures are produced by `plot_bloom_ioi.py`.

Useful next analyses are additional discovery seeds, corruption choices, MLP-freezing sensitivity and expanded node sets when fidelity is low. If these results guide selection changes, use a fresh held-out seed for the final assessment.

## Reproduce this configuration

```bash
python English_Task/discover_bloom_ioi.py --offline \
  --model bigscience/bloom-560m --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
  --stages baseline,direct,paths,functions,validate \
  --seed 42 --n-per-pattern 200 --discovery-n 50 --validation-n 200 \
  --batch-size 8 --corruption abc --top-k 8 --min-effect 0.02 \
  --routing-threshold 0.1 --mean-donors 8 --random-trials 3 \
  --probe-samples 32 --probe-length 32 --head-roles IO,S1,S1+1,S2,END \
  --output-dir English_Task/english_results/reproduction
```
