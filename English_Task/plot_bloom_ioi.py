#!/usr/bin/env python3
"""Export signed patching heatmaps and held-out circuit performance."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, help='Default: patching_summary.png beside results')
    parser.add_argument('--dependency-dir', type=Path, help='Optional separate directory containing plotting packages')
    args = parser.parse_args()
    if args.dependency_dir:
        sys.path.insert(0, str(args.dependency_dir.resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    result = json.loads(args.results.read_text())
    config = result['config']
    layers = max(config['layers']) + 1
    # All tested heads appear in the direct sweep, including unselected heads.
    heads = max(r['node']['head'] for r in result['direct']) + 1
    panels = [('direct', 'Head at END → logits'),
              ('query_paths', 'END → downstream Q at END'),
              ('value_paths', 'S2 → downstream V at S2'),
              ('key_paths', 'S1+1 → induction K at S1+1')]
    fig, axes = plt.subplots(1, 4, figsize=(15, 7), constrained_layout=True, sharey=True)
    fig.suptitle('BLOOM English IOI: signed noising path effects\n'
                 f'{2 * config["discovery_n"]} discovery prompts; untested cells are gray', fontsize=14)
    for ax, (stage, title) in zip(axes, panels):
        records = result.get(stage, [])
        if isinstance(records, dict):
            records = records['records']
        grid = np.full((layers, heads), np.nan)
        for record in records:
            node = record['node']
            grid[node['layer'], node['head']] = record['effects']['all']['delta']['mean']
        finite = np.abs(grid[np.isfinite(grid)])
        limit = max(float(finite.max()) if finite.size else 0, 0.01)
        cmap = plt.get_cmap('RdBu_r').copy()
        cmap.set_bad('#dddddd')
        plotted = ax.imshow(grid, cmap=cmap, vmin=-limit, vmax=limit, aspect='auto', interpolation='nearest')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Head')
        ax.set_xticks(range(0, heads, 2))
        ax.set_yticks(range(0, layers, 2))
        fig.colorbar(plotted, ax=ax, shrink=0.65, label='Patched − clean logit difference')
        selected = (result.get('output_candidates', []) if stage == 'direct'
                    else result.get(stage, {}).get('selected', []))
        for node in selected:
            ax.plot(node['head'], node['layer'], 's', markerfacecolor='none', markeredgecolor='black', markersize=7)
    axes[0].set_ylabel('Layer (zero-based)')
    output = args.output or args.results.parent / 'patching_summary.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)
    if 'validate' in result:
        summaries = result['validate']['summaries']
        labels = ['full', 'empty_attention', 'candidate_only', 'candidate_removed']
        labels += [k for k in summaries if k.startswith('random_')]
        means = [summaries[k]['all']['logit_diff']['mean'] for k in labels]
        errors = [summaries[k]['all']['logit_diff']['standard_error'] for k in labels]
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax.bar(range(len(labels)), means, yerr=errors, capsize=4,
               color=['#325c9c', '#aaa', '#438558', '#a96749'] + ['#aaa'] * (len(labels)-4))
        ax.set_xticks(range(len(labels)), [k.replace('_', '\n') for k in labels])
        ax.set_ylabel('Mean logit(IO) − logit(S), ± standard error')
        ax.set_title(f'BLOOM English IOI: {2 * config["validation_n"]} held-out prompts\n'
                     'Attention nodes ablated; embeddings and MLPs retained')
        ax.axhline(0, color='black', linewidth=0.5)
        fig.savefig(output.parent / 'validation_summary.png', dpi=180)
        fig.savefig(output.parent / 'validation_summary.pdf')
        plt.close(fig)
    print(f'Saved figures to {output.parent.resolve()}')


if __name__ == '__main__':
    main()
