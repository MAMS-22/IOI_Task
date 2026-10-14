#!/usr/bin/env python3
"""Write a compact, reviewable Markdown report from a discovery artifact."""
import argparse
import json
from pathlib import Path
import shlex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    config = result['config']
    lines = ['# English BLOOM IOI measurements', '',
        f'Model: `{config["model"]}`. Discovery seed: {config["seed"]}. '
        f'Discovery: {2*config["discovery_n"]} prompts; held-out evaluation: {2*config["validation_n"]} prompts.', '',
        'This is the English path-patching branch of the supplied paper. '
        'The lexical pools and sampling are not verified as the authors’ exact dataset. '
        'Candidate nodes are hypotheses, not automatically proven functional heads.', '',
        'Completed stages: ' + ', '.join(result['completed_stages']) + '.', '',
        '## Baseline', '', '| Pattern | Examples | IO > S | IO top-1 | Mean logit difference ± SE |',
        '|---|---:|---:|---:|---:|']
    for pattern, value in result.get('baseline', {}).get('clean', {}).items():
        d=value['logit_diff']
        lines.append(f'| {pattern} | {d["n"]} | {value["accuracy"]:.1%} | {value["zero_rank_rate"]:.1%} | {d["mean"]:.3f} ± {d["standard_error"]:.3f} |')
    lines.extend(['', '## Held-out attention-node validation', '',
        'Embeddings and MLPs remain active in every condition. Outside-circuit attention '
        'nodes are replaced by the mean of independent disjoint-ABC donors in the same context.'])
    validation=result.get('validate')
    if validation:
        lines.extend(['', '| Condition | IO > S | IO top-1 | Mean logit difference ± SE |', '|---|---:|---:|---:|'])
        for label, summaries in validation['summaries'].items():
            if label.startswith('leave_out_'):
                continue
            value=summaries['all'];d=value['logit_diff']
            lines.append(f'| {label} | {value["accuracy"]:.1%} | {value["zero_rank_rate"]:.1%} | {d["mean"]:.3f} ± {d["standard_error"]:.3f} |')
        lines.extend(['', '| Pattern | Attention-dependent gap recovered | Full-model logit difference retained |', '|---|---:|---:|'])
        for pattern,value in validation['fidelity'].items():
            recovery=value['normalized_attention_recovery'];retained=value['fraction_of_full_logit_diff']
            a='undefined' if recovery is None else f'{recovery:.1%}'
            b='undefined' if retained is None else f'{retained:.1%}'
            lines.append(f'| {pattern} | {a} | {b} |')
        lines.extend(['', 'Accuracy alone can conceal lost logit margin. Compare recovered margin '
            'against random controls and both name-order patterns. This establishes only '
            'attention-node fidelity under this ablation distribution, not full-model completeness.'])
    else:
        lines.extend(['', 'Held-out validation has not completed.'])
    lines.extend(['', '## Candidate nodes and functional evidence', '',
        'Duplicate/induction probe columns are repeated-sequence score minus non-repeating control score. '
        'Previous-token behavior is the absolute score on the non-repeating control, where these heads should also work. '
        'Attribution is measured at END even when a retained node is at another position.', '',
        '| Head | Retained position | END IO−S attribution | END→IO | END→S2 | Duplicate probe Δ | Induction probe Δ | Previous-token score |',
        '|---|---|---:|---:|---:|---:|---:|---:|'])
    evidence={(x['layer'],x['head']):x for x in result.get('functions',result.get('selection_evidence',[]))}
    for node in result['candidate_nodes']:
        x=evidence.get((node['layer'],node['head']))
        if x:
            ioi=x['ioi'];probe=x['random_tokens']
            differences=[probe[k+'_repeated']['mean']-probe[k+'_unique']['mean'] for k in ('duplicate','induction')]
            differences.append(probe['previous_unique']['mean'])
            values=[ioi['end_logit_diff_attribution']['mean'],ioi['attention_END_to_IO']['mean'],ioi['attention_END_to_S2']['mean'],*differences]
            lines.append(f'| {node["layer"]}.{node["head"]} | {node["role"]} | ' + ' | '.join(f'{v:+.3f}' for v in values) + ' |')
        else:
            lines.append(f'| {node["layer"]}.{node["head"]} | {node["role"]} | — | — | — | — | — | — |')
    lines.extend(['', f'{len(result.get("confirmed_edges",[]))} individual sender/receiver paths measured. '
        'Signs, paired standard errors and receiver attention changes are in `results.json`. '
        'The graph is in `candidate_circuit.dot`; path and validation figures are produced by `plot_bloom_ioi.py`.', '',
        'Useful next analyses are additional discovery seeds, corruption choices, MLP-freezing '
        'sensitivity and expanded node sets when fidelity is low. If these results guide '
        'selection changes, use a fresh held-out seed for the final assessment.', '',
        '## Reproduce this configuration', '', '```bash',
        'python English_Task/discover_bloom_ioi.py --offline \\',
        f'  --model {shlex.quote(config["model"])} --layers {",".join(str(l) for l in config["layers"])} \\',
        '  --stages ' + ','.join(result['completed_stages']) + ' \\',
        f'  --seed {config["seed"]} --n-per-pattern {config["n_per_pattern"]} --discovery-n {config["discovery_n"]} --validation-n {config["validation_n"]} \\',
        f'  --batch-size {config["batch_size"]} --corruption {config["corruption"]} --top-k {config["top_k"]} --min-effect {config["min_effect"]} \\',
        f'  --routing-threshold {config.get("routing_threshold",0)} --mean-donors {config["mean_donors"]} --random-trials {config["random_trials"]} \\',
        f'  --probe-samples {config["probe_samples"]} --probe-length {config["probe_length"]} --head-roles {config["head_roles"]} \\',
        ('  --freeze-mlps \\' if config['freeze_mlps'] else None),
        ('  --no-bos \\' if config['no_bos'] else None),
        ('  --circuit ' + shlex.quote(config['circuit']) + ' \\' if config.get('circuit') else None),
        '  --output-dir English_Task/english_results/reproduction', '```', ''])
    output=args.results.parent/'report.md'
    output.write_text('\n'.join(line for line in lines if line is not None))
    print(output.resolve())


if __name__ == '__main__':
    main()
