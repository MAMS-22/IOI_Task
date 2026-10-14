#!/usr/bin/env python3
"""Generate the meeting brief and scientific figures from frozen experiments."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import csv
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'english_results/meeting_reproduction')
    parser.add_argument('--dependency-dir',type=Path,default=Path('/tmp/ioi-plot-deps'))
    args=parser.parse_args()
    if args.dependency_dir.exists():
        sys.path.insert(0,str(args.dependency_dir))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    root=args.output_dir
    def read(name):
        value=json.loads((root/(name+'.json')).read_text())
        if name!='manifest' and value.get('status')!='complete':
            raise ValueError(f'Incomplete experiment: {name}')
        return value
    manifest=read('manifest')
    assembled=read('assembled')
    pruned=read('prune')
    result=read('evaluate_discovered_test')
    functions=read('functions')
    mechanisms=read('mechanisms')
    knockouts=read('knockouts')
    specificity=read('specificity')
    hook_validation=read('hook_validation')
    reference=json.loads((Path(__file__).resolve().parent/'paper_bloom_reference.json').read_text())
    def key(node):
        return node['layer'],node['head'],node['role']
    paper={ (l,h,group['role']) for group in reference['groups'].values() for l,h in group['heads'] }
    broad={key(n) for n in assembled['candidate_nodes']}
    core={key(n) for n in assembled['functional_core_nodes']}
    sparse={key(n) for n in pruned['candidate_nodes']}
    direct={key(n) for n in read('direct')['selected_nodes']}
    if result['pruned_circuit_sha256']!=hashlib.sha256((root/'prune.json').read_bytes()).hexdigest():
        raise ValueError('Final evaluation does not match the frozen pruned mask')
    prompt_sets=[{ex['prompt'] for ex in manifest[split]} for split in ('discovery','confirmation','test')]
    if any(prompt_sets[i]&prompt_sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Task splits overlap')
    full_prompt_ids=[r['example_id'] for r in result['evaluation']['rows']['full']]
    if any([r['example_id'] for r in values]!=full_prompt_ids for values in result['evaluation']['rows'].values()):
        raise ValueError('Paired comparison rows are misaligned')
    summary=result['evaluation']['summaries']
    rows=result['evaluation']['rows']
    fidelity=result['evaluation']['fidelity']
    rng=np.random.default_rng(20261014200)
    full=np.array([r['logit_diff'] for r in rows['full']])
    context_rows={}
    context_templates={}
    for index,row in enumerate(rows['full']):
        context=row['example_id'].rsplit('_',1)[0]
        context_rows.setdefault(context,[]).append(index)
        context_templates[context]=row['template_id']
    if any(len(indices)!=2 for indices in context_rows.values()):
        raise ValueError('Expected paired ABBA/BABA contexts for cluster bootstrap')
    stratified=[]
    for template in sorted(set(context_templates.values())):
        contexts=[context_rows[c] for c in sorted(context_rows) if context_templates[c]==template]
        indices=np.asarray(contexts)
        selected=rng.integers(0,len(indices),size=(2000,len(indices)))
        stratified.append(indices[selected].reshape(2000,-1))
    draw=np.concatenate(stratified,axis=1)
    def wilson(rate,n):
        z=1.96
        center=(rate+z*z/(2*n))/(1+z*z/n)
        radius=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
        return float(center-radius),float(center+radius)
    ci={}
    for name,values in rows.items():
        margin=np.array([r['logit_diff'] for r in values])
        ratios=margin[draw].mean(1)/full[draw].mean(1)
        deltas=(margin-full)[draw].mean(1)
        ci[name]=dict(ratio_bootstrap_95=list(np.quantile(ratios,[.025,.975])),
            paired_margin_delta_bootstrap_95=list(np.quantile(deltas,[.025,.975])),
            accuracy_context_bootstrap_95=list(np.quantile(np.asarray([r['correct'] for r in values])[draw].mean(1),[.025,.975])),
            top1_context_bootstrap_95=list(np.quantile(np.asarray([r['zero_rank'] for r in values])[draw].mean(1),[.025,.975])),
            accuracy_wilson_95=list(wilson(summary[name]['all']['accuracy'],len(values))),
            top1_wilson_95=list(wilson(summary[name]['all']['zero_rank_rate'],len(values))))
    knockout_ci={}
    for label in knockouts['incompleteness']:
        crows=knockouts['evaluation']['rows']['C_without_'+label]
        mrows=knockouts['evaluation']['rows']['M_without_'+label]
        if [r['example_id'] for r in crows]!=full_prompt_ids or [r['example_id'] for r in mrows]!=full_prompt_ids:
            raise ValueError('Knockout rows do not align with final-test bootstrap contexts')
        deltas=np.asarray([a['logit_diff']-b['logit_diff'] for a,b in zip(crows,mrows)])
        knockout_ci[label]=list(np.quantile(deltas[draw].mean(1),[.025,.975]))
    names=['full','legacy_33_nodes','paper_reference','confirmed_broad','functional_core','discovered']
    labels=['Full model','Previous 33 nodes','Paper 38 (reference)','Confirmed broad','Functional core','Pruned discovery']
    counts={name:len(result['conditions'].get(name,[])) for name in result['conditions']}
    counts['full']=24*16*len(manifest['test'][0]['clean_ids'])
    groups={}
    for label,group in reference['groups'].items():
        wanted={(l,h,group['role']) for l,h in group['heads']}
        groups[label]=dict(paper_count=len(wanted),broad_match=len(wanted&broad),core_match=len(wanted&core),
            sparse_match=len(wanted&sparse),missing_from_broad=[list(n) for n in sorted(wanted-broad)],
            missing_from_sparse=[list(n) for n in sorted(wanted-sparse)])
    provenance=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        reporter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bootstrap_seed=20261014200,bootstrap_context_count=len(context_rows),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.json')) if p.name!='meeting_metrics.json'},
        circuit_counts=dict(broad_nodes=len(broad),broad_unique_heads=len({n[:2] for n in broad}),
            core_nodes=len(core),core_unique_heads=len({n[:2] for n in core}),
            sparse_nodes=len(sparse),sparse_unique_heads=len({n[:2] for n in sparse})),
        paper_overlap=dict(groups=groups,broad_total=len(paper&broad),core_total=len(paper&core),sparse_total=len(paper&sparse)),
        intervals=ci,knockout_intervals=knockout_ci,
        interval_policy='2,000 paired ABBA/BABA context-cluster bootstrap resamples, stratified within fixed templates. Shown accuracy/top-1 intervals use this bootstrap; Wilson prompt intervals are auxiliary independence heuristics. Zero observed errors yield degenerate bootstrap intervals, not proof of population perfection. Template-bank and discovery-seed uncertainty are not estimated.')
    (root/'meeting_metrics.json').write_text(json.dumps(provenance,indent=2)+'\n')
    def pct(x): return f'{100*x:.1f}%'
    def table():
        text=['| Condition | Retained nodes | IO > S | Vocabulary top-1 | Mean IO−S | Margin/full |','|---|---:|---:|---:|---:|---:|']
        for name,label in zip(names,labels):
            s=summary[name]['all']
            ratio=1 if name=='full' else fidelity[name]['all']['fraction_of_full_logit_diff']
            count='all' if name=='full' else str(counts[name])
            text.append(f"| {label} | {count} | {pct(s['accuracy'])} | {pct(s['zero_rank_rate'])} | {s['logit_diff']['mean']:.3f} | {pct(ratio)} |")
        return '\n'.join(text)
    metric=summary['discovered']['all']
    ratio=fidelity['discovered']['all']['fraction_of_full_logit_diff']
    ratio_ci=ci['discovered']['ratio_bootstrap_95']
    random_ratios=[fidelity[f'matched_random_{i}']['all']['fraction_of_full_logit_diff'] for i in range(5)]
    counter=mechanisms['counterfactual_summaries']
    max_knockout=max(knockouts['incompleteness'].items(),key=lambda pair:pair[1]['all']['absolute_mean_gap'])
    routes=[]
    edge_tests=[]
    for target in ('positive_output_q','negative_output_q','subject_end_q','subject_end_v','subject_end_k','induction_k'):
        artifact=read('path_'+target)
        routes.append(f"| {target} | {len(artifact['discovery_selected_nodes'])} | {len(artifact['selected_nodes'])} | {len(artifact['individual_edges'])} |")
        for edge in artifact['individual_edges']:
            per_pattern=[]
            for pat in ('ABBA','BABA'):
                effect=edge['effects'][pat]['delta']
                m,se=effect['mean'],effect['standard_error']
                probability=math.erfc(abs(m)/se/math.sqrt(2)) if se>0 else (0. if m else 1.)
                per_pattern.append((probability,pat,m,se))
            probability,pat,m,se=min(per_pattern)
            # Two pattern tests: Bonferroni before the global descriptive BH screen.
            edge_tests.append(dict(edge=edge,target=target,p=min(1,2*probability),pattern=pat,mean=m,se=se))
    edge_tests.sort(key=lambda e:e['p'])
    cutoff=0.
    for i,edge in enumerate(edge_tests,1):
        if edge['p']<=.05*i/max(len(edge_tests),1): cutoff=edge['p']
    supported_edges=[]
    for test in edge_tests:
        e=test['edge']
        sender=key(e['sender'])
        receiver=key(e['receiver'])
        output_role='END' if test['target']!='induction_k' else 'S2'
        mapped_receiver=(receiver[0],receiver[1],output_role)
        if sender in sparse and mapped_receiver in sparse and test['p']<=cutoff and abs(test['mean'])>=.001:
            supported_edges.append(dict(sender=list(sender),receiver_output=list(mapped_receiver),
                receiver_input=list(receiver),component=e['component'],pattern=test['pattern'],
                delta=test['mean'],standard_error=test['se'],approximate_p=test['p'],route=test['target']))
    circuit_export=dict(nodes=[dict(n,**{'published_position_match':key(n) in paper}) for n in pruned['candidate_nodes']],
        edges=supported_edges,
        edge_policy='Descriptive edge screen on independent confirmation data: two patterns Bonferroni, global BH .05, effect >=.001. Only frozen-mask endpoints shown. Receiver input position can differ from its retained output position. This subset is not a complete computation graph.',
        pruning_mask_sha256=result['pruned_circuit_sha256'])
    (root/'compact_circuit.json').write_text(json.dumps(circuit_export,indent=2)+'\n')
    with (root/'compact_circuit_edges.csv').open('w') as file:
        writer=csv.writer(file)
        writer.writerow(['sender_layer','sender_head','sender_position','receiver_layer','receiver_head','receiver_retained_output_position','receiver_input_position','component','pattern','delta','standard_error','approximate_p'])
        for e in supported_edges:
            writer.writerow([*e['sender'],*e['receiver_output'],e['receiver_input'][2],e['component'],e['pattern'],e['delta'],e['standard_error'],e['approximate_p']])
    role_table='\n'.join(['| Paper family | Paper nodes | Broad recovered | Core recovered | Pruned recovered |','|---|---:|---:|---:|---:|']+
        [f"| {name} | {g['paper_count']} | {g['broad_match']} | {g['core_match']} | {g['sparse_match']} |" for name,g in groups.items()])
    brief=f'''# English BLOOM IOI reproduction: meeting brief

The validated pipeline now discovers and prunes an English IOI attention circuit independently of the paper's head IDs. On **1,000 fresh prompts (500 matched contexts)**, the frozen {len(sparse)}-node circuit achieves **{pct(metric['accuracy'])} IO-vs-subject accuracy** and retains **{pct(ratio)} of the full model's mean logit difference** (context-cluster bootstrap 95% interval {pct(ratio_ci[0])}–{pct(ratio_ci[1])}). Its vocabulary top-1 accuracy is **{pct(metric['zero_rank_rate'])}**. Report all three metrics together.

This is a method and functional reproduction for BLOOM-560m in English. It is not an exact reproduction of the authors' corpus, all languages, information-flow results, or a proof of completeness/minimality.

## Show these three things

1. Open [meeting_figures.pdf](meeting_figures.pdf), pages 1–2: a controlled comparison on the same untouched test prompts and independent ABC ablation means. Page 7 displays a measured subset of receiver-input paths in the compact mask.
2. Show page 3: the broad search recovered {len(paper&broad)}/38 paper-position nodes within 443 candidates, including {len({n for n in paper if n[2]=='END'}&direct)}/20 final-position heads within 58 direct candidates. The compact mask retains {len(paper&sparse)}/38. Paper IDs were loaded only for evaluation, not used to choose heads. Coverage alone is not a sparse graph reproduction.
3. Show pages 4–5: independent copying/duplicate/induction probes and token/order interventions. For example, patching the subject-focused heads from the name-swapped input changes the mean original-target margin from {counter['original']['logit_diff']['mean']:.2f} to {counter['name_swap']['logit_diff']['mean']:.2f}; changing order gives {counter['order_swap']['logit_diff']['mean']:.2f}. These are causal group interventions, not a claim that each head has a fully isolated role.

## Controlled final-test results

{table()}

All circuit conditions retain embeddings and MLPs. A node is a head at a semantic token position; unique heads and nodes are different counts. The pruned mask contains {len({n[:2] for n in sparse})} unique heads. The paper's 38-head mask here uses inferred positional retention conventions, not an author-supplied ablation mask. The paper reports full-model English IO-vs-S accuracy of 100%/100% and top-1 of 96%/91% for normal/flipped prompts; our ABBA/BABA labels are not verified to map to those labels.

Pruning optimized IO-vs-S accuracy and mean margin, not vocabulary top-1 or completeness. It removes all three published negative heads. Its stronger intact margin than the published-reference mask does not establish a more accurate functional graph.

## Three-minute explanation

“The previous implementation found some of the expected heads but discarded weak paths with a fixed top-k rule and pooled the two prompt orderings. I replaced that with separate paired screens for ABBA and BABA, independent confirmation, separate positive/negative/subject receiver groups, and query/key/value path tests. I also replaced evaluation-derived ablation donors with 3,000 independent, template-conditioned ABC reference prompts.

“I first found a broad causal mask, then greedily pruned it using only 200 development prompts. I froze that mask before evaluating 1,000 new prompts. The old circuit, paper reference, new circuit, functional core and five matched random masks are all tested under identical controls. The headline is {pct(metric['accuracy'])} name-choice accuracy and {pct(ratio)} retained margin, versus {pct(fidelity['legacy_33_nodes']['all']['fraction_of_full_logit_diff'])} for the old mask. The random-mask margin range is {pct(min(random_ratios))}–{pct(max(random_ratios))}.

“We also reproduce several functional behaviors and causal sensitivity to the repeated name and its order. I would call this a stronger English method reproduction, with remaining gaps in the exact functional graph, author-data equivalence and circuit completeness.”

## Questions to expect

- **Is the circuit complete?** No. It fails the matched C\\K versus M\\K stress tests: the largest observed absolute mean gap is {max_knockout[1]['all']['absolute_mean_gap']:.3f} logits ({max_knockout[0]}), showing substantial unmodeled compensation. Page 6 shows the measured gaps. Strong intact name-choice performance does not imply a complete causal graph.
- **Did you just use the published heads?** No. The selection code uses effects, routing and random probes. The published mask is a clearly labeled benchmark. Pruning may remove a published head when other routes compensate.
- **Why more than 38 heads?** The causal search admits generic support routes, while the paper's circuit groups heads by function. Greedy pruning improves sparsity but does not establish the smallest sufficient mask. The functional core is a separate predefined sensitivity analysis.
- **Does 100% name-choice accuracy mean the same output as the model?** No. IO-vs-S checks only two names. Vocabulary top-1 and retained margin are reported separately.
- **What changed besides the algorithm?** The lexical pools now follow the vendored Wang dataset and are filtered to single BLOOM tokens: 90 names, eight places and seven objects, with 15 templates per ordering. Because the corpus differs, all old/new/reference masks were re-evaluated on one shared fresh split.
- **Is this the paper's information flow?** No. Attention probabilities, causal patching and OV capability tests are reported. Ferrando/Voita flow and cross-language comparison remain unimplemented.

## Next experiments after the meeting

Obtain the authors' exact English corpus and run protocol; isolate remaining role assignments with controlled receiver-input interventions; examine the largest matched-knockout discrepancies; repeat discovery across seeds and templates; then reproduce the information-flow calculation before adding languages. Preserve this final test as a sealed benchmark: later changes need a new test split.

See [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md) for methods, uncertainty, commands and raw checkpoints.
'''
    (root/'MEETING_BRIEF.md').write_text(brief)
    detailed=f'''# English BLOOM IOI reproduction report

Generated {provenance['created_utc']}. Model: `{manifest['protocol']['model']}`. No weights were trained or fine-tuned.

## Final test

{table()}

Accuracy means IO logit > S logit. Top-1 means IO is first across the complete vocabulary. Margin/full is the ratio of mean logit differences; it is not bounded to [0,1]. Empty-adjusted recovery is stored separately in `evaluate_discovered_test.json`. Shown final-test intervals resample the 500 matched ABBA/BABA contexts as clusters, stratified within the 15 fixed templates, preserving both model/circuit and ordering dependence. They do not estimate template-bank or discovery-seed uncertainty. Zero observed errors give a degenerate bootstrap interval and do not prove population perfection. Auxiliary Wilson intervals are stored as prompt-independence heuristics, not used in figures.

| Ordering | Full accuracy | Full top-1 | Pruned accuracy | Pruned top-1 | Pruned margin/full |
|---|---:|---:|---:|---:|---:|
'''
    for pat in ('ABBA','BABA'):
        f,s=summary['full'][pat],summary['discovered'][pat]
        detailed+=f"| {pat} | {pct(f['accuracy'])} | {pct(f['zero_rank_rate'])} | {pct(s['accuracy'])} | {pct(s['zero_rank_rate'])} | {pct(fidelity['discovered'][pat]['fraction_of_full_logit_diff'])} |\n"
    detailed+=f'''
## Protocol and validation

- Discovery: 400 prompts; confirmation/development: 200; sealed final test: 1,000; ABC mean bank: 3,000 (100 per template and ordering). Task prompts are explicitly disjoint, and previously evaluated prompts were excluded.
- Seed: {manifest['protocol']['seed']}; the manifest stores every prompt, token ID, answer token and semantic index. Explicit BOS and float32 are used. The vocabulary comes from the vendored Redwood/Wang dataset and is filtered for leading-space single BLOOM tokens.
- Discovery uses two-sided paired mean/SE normal-approximation tests with BH screening separately for ABBA and BABA. The accepted union avoids ordering cancellation. Direct discovery floor: .02 logits; upstream/confirmation floor: .001; FDR screen: .05. This screen is not proof of functional relevance and does not control a global error rate over all sequential stages.
- Path patching freezes other attention outputs, corrupts a sender, allows intervening MLP/LayerNorm recomputation, records receiver Q/K/V and replays that input into a fresh clean run. The intervention is nonlinear and uses actual BLOOM hooks.
- Six real-model checks passed: actual Q/K/V weight layouts, Z/result equivalence, identical-donor path identities, exact frozen-MLP residual/LN reconstruction, affine/centered readout, and all-attention retention identity. Fourteen CPU regression tests passed.
- `hook_validation.json` records the executed assertions and numerical errors on {hook_validation['n_prompts']} diagnostic prompts. The largest recorded absolute error is {max(c['max_abs_error'] for c in hook_validation['checks'].values()):.3g}, checked with absolute tolerance {hook_validation['atol']:.3g} and relative tolerance {hook_validation['rtol']:.3g}. These sanity checks do not cover every prompt or every Q/K/V layer.
- Confirmed broad mask: {len(broad)} nodes/{len({n[:2] for n in broad})} unique heads. Functional core: {len(core)} nodes/{len({n[:2] for n in core})} heads. Pruned: {len(sparse)} nodes/{len({n[:2] for n in sparse})} heads. Semantic position retention is explicit. Embeddings and MLPs remain present.
- Pruning uses only development prompts: marginal ranking followed by blocks of 16/4/1, with fixed per-ordering margin and accuracy targets stored in `prune.json`. The budget flag is `{pruned['budget_reached']}`. Finite greedy pruning does not prove minimality. The mask hash in final evaluation must match `prune.json`.

The development margin targets were ABBA {pruned['targets']['ABBA']['margin']:.3f} ({pct(pruned['targets']['ABBA']['margin']/pruned['full']['ABBA']['logit_diff']['mean'])} of full) and BABA {pruned['targets']['BABA']['margin']:.3f} ({pct(pruned['targets']['BABA']['margin']/pruned['full']['BABA']['logit_diff']['mean'])} of full). ABBA starts below the nominal 90% target, so its floor is the initial broad-mask margin. The algorithm does not claim to attain 90% for both patterns. Vocabulary top-1 and completeness were evaluated but not optimized.

## Search coverage

| Receiver route | Discovery senders | Independently confirmed | Individual edge tests |
|---|---:|---:|---:|
'''+'\n'.join(routes)+f'''

Individual edge effects are stored without treating every tested edge as established. Weak group effects and individual effects need not agree under interaction or cancellation. Functional labels are provisional and may differ from the paper even for a matching head.

{role_table}

Overlap is measured at the paper's inferred semantic positions. The broad screen recovers {len(paper&broad)}/38 paper nodes, but also many extras. The pruned mask recovers {len(paper&sparse)}/38. Neither head overlap alone nor an ablation score proves the paper's exact graph.

## Functional evidence

`functions.json` stores separate IO and S readout contributions, routing attention, repeated/unique random-token probes and positive/negative OV capability scores over all 90 names. `mechanisms.json` independently repeats functional measurements on 200 confirmation prompts and stores per-example attention/projection scatter values and correlations.

OV probes use the name residual after block 0, each head's own LayerNorm, W_V/W_O without projection biases, and the actual final LN/unembedding. They are capability tests, not direct evidence of causal usage. Random-token controls use BLOOM vocabulary tokens; these are an adapted diagnostic, not an exact replication of the authors' language-specific sampling protocol.

Joint activation replacement in the subject-focused END heads gives these original-target mean margins: original {counter['original']['logit_diff']['mean']:.3f}; name swap {counter['name_swap']['logit_diff']['mean']:.3f}; initial-order swap {counter['order_swap']['logit_diff']['mean']:.3f}; both {counter['both_swap']['logit_diff']['mean']:.3f}. Order changes alter which first name is repeated, while name changes preserve that location. Because BLOOM uses ALiBi, this experiment does not copy a learned position embedding. The intervention is joint END-output replacement, not exclusive isolation of each signal along a Q path.

The supplemental subject-only corruption changes S2 to IO and leaves both initial names fixed. It supports {len(specificity['selected_nodes'])} direct END candidates. It does not alter the primary circuit. Its complete effects are in `specificity.json`.

The compact mask retains {len(paper&sparse)}/38 published nodes and removes all three published negative heads. Its higher retained mean margin than the paper-reference mask is therefore not evidence of a more accurate functional graph: removing negative contributions can increase the margin. The broad mask is a coverage result, not a sparse recovery of 38 heads.

## Completeness stress tests

Matched ablations compare C\\K with M\\K on {len(knockouts['evaluation']['rows']['full'])} {knockouts['split']} prompts. K includes positional/functional groups and ten fixed random subsets. The largest pooled gap is {max_knockout[1]['all']['absolute_mean_gap']:.3f} logits, demonstrating that the compact mask is incomplete under these tests. Passing finite tests would not establish exhaustive completeness, and greedy pruning does not establish conditional minimality. Group masks reflect our provisional roles, not a one-to-one validation of paper annotations.

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

Each checkpoint records source hashes, configuration, timing and dependency versions. Versions: `{json.dumps(result['versions'])}`. `meeting_metrics.json` records input artifact SHA-256 hashes and statistical intervals. `meeting_figures.pdf` contains seven scientific figure pages; PNG counterparts are in `figures/`. `compact_circuit.json` and `compact_circuit_edges.csv` export the mask and the supported subset of measured paths. `MEETING_BRIEF.md` is the short talk track. Preserve these artifacts and the final mask hashes when reporting results.
'''
    (root/'REPRODUCTION_REPORT.md').write_text(detailed)
    figure_dir=root/'figures'
    figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#334155','#ef4444','#8b5cf6','#94a3b8','#f59e0b','#0d9488']
    def finish(fig,pdf,name,title,caption):
        fig.suptitle(title,fontsize=16)
        fig.text(.04,.02,caption,fontsize=8,wrap=True)
        fig.tight_layout(rect=(0,.07,1,.94))
        pdf.savefig(fig)
        fig.savefig(figure_dir/(name+'.png'),dpi=180)
        plt.close(fig)
    with PdfPages(root/'meeting_figures.pdf') as pdf:
        fig,axes=plt.subplots(1,3,figsize=(14,6))
        x=np.arange(len(names))
        for ax,field,title in zip(axes,['accuracy','zero_rank_rate','margin'],['IO > S accuracy','Vocabulary top-1','Mean margin / full model']):
            vals=[summary[n]['all'][field]*100 if field!='margin' else (1 if n=='full' else fidelity[n]['all']['fraction_of_full_logit_diff'])*100 for n in names]
            ax.bar(x,vals,color=colors)
            if field=='margin':
                error=np.array([[vals[i]-100*ci[n]['ratio_bootstrap_95'][0],100*ci[n]['ratio_bootstrap_95'][1]-vals[i]] for i,n in enumerate(names)]).T
            else:
                interval_field='accuracy_context_bootstrap_95' if field=='accuracy' else 'top1_context_bootstrap_95'
                error=np.array([[vals[i]-100*ci[n][interval_field][0],100*ci[n][interval_field][1]-vals[i]] for i,n in enumerate(names)]).T
            ax.errorbar(x,vals,yerr=np.maximum(error,0),fmt='none',ecolor='black',capsize=2)
            ax.set_xticks(x,labels,rotation=40,ha='right')
            ax.set_title(title)
            ax.set_ylabel('%')
            for i,v in enumerate(vals): ax.text(i,v+2,f'{v:.1f}',ha='center',fontsize=8)
        finish(fig,pdf,'01_performance','Frozen masks: controlled final test, 1,000 prompts',
            'Same test set and 3,000 independent ABC reference prompts. Embeddings and MLPs retained. Error bars: 95% matched-context cluster bootstrap, stratified by template. Zero-error bars are degenerate. Paper mask uses inferred positions.')
        fig,axes=plt.subplots(1,2,figsize=(13,6))
        order_names=['legacy_33_nodes','paper_reference','functional_core','discovered']
        order_labels=['Previous 33','Paper reference','Functional core','Pruned discovery']
        for j,pat in enumerate(('ABBA','BABA')):
            vals=[fidelity[n][pat]['fraction_of_full_logit_diff']*100 for n in order_names]
            axes[0].bar(np.arange(4)+(j-.5)*.36,vals,width=.36,label=pat)
        axes[0].set_xticks(range(4),order_labels,rotation=25,ha='right')
        axes[0].set_ylabel('Mean margin / full (%)')
        axes[0].axhline(100,color='gray',ls='--'); axes[0].legend()
        axes[1].bar(range(5),np.array(random_ratios)*100,color='#94a3b8',label='Matched random masks')
        axes[1].axhline(ratio*100,color='#0d9488',lw=2,label=f'Pruned discovery ({len(sparse)} nodes)')
        axes[1].set_xticks(range(5),[f'Random {i+1}' for i in range(5)])
        axes[1].set_ylabel('Mean margin / full (%)'); axes[1].legend()
        finish(fig,pdf,'02_ordering_controls','Ordering cancellation and matched random controls',
            'Five random masks match the primary circuit node counts at each layer and semantic role. Finite controls, not a p-value. Ratios above 100% are possible and are not clipped.')
        fig,axes=plt.subplots(1,2,figsize=(13,6))
        group_names=list(groups)
        for j,(field,label) in enumerate([('paper_count','Paper'),('broad_match','Broad'),('sparse_match','Pruned')]):
            axes[0].bar(np.arange(len(groups))+(j-1)*.25,[groups[g][field] for g in group_names],width=.25,label=label)
        axes[0].set_xticks(range(len(groups)),group_names,rotation=30,ha='right'); axes[0].set_ylabel('Nodes'); axes[0].legend()
        matrix=np.zeros((24,16))
        for l,h,role in sparse: matrix[l,h]+=1
        image=axes[1].imshow(matrix,origin='lower',aspect='auto',cmap='Greens',vmin=0)
        for l,h,role in paper: axes[1].plot(h,l,'s',mfc='none',mec='#8b5cf6',ms=8,mew=1)
        axes[1].set_xlabel('Head'); axes[1].set_ylabel('Layer'); axes[1].set_title(f'Pruned mask: {len(sparse)} nodes / {len({n[:2] for n in sparse})} heads')
        fig.colorbar(image,ax=axes[1],label='Retained semantic positions')
        finish(fig,pdf,'03_overlap','Independent discovery versus published functional families',
            'Purple squares mark paper heads; green intensity counts retained positions. Overlap is descriptive. Pruning used development performance, not published head IDs. A shared head is not proof of the same functional role.')
        evidence={(e['layer'],e['head']):e for e in functions['evidence']}
        copy={(e['layer'],e['head']):e for e in functions['ov_copy']}
        fig,axes=plt.subplots(1,3,figsize=(14,6))
        for ax,group_name,probe in zip(axes[:2],['duplicate','induction'],['duplicate','induction']):
            group=next(g for name,g in reference['groups'].items() if probe in name.lower())
            head_ids=[tuple(pair) for pair in group['heads']]
            for j,cond in enumerate(('unique','repeated')):
                vals=[evidence[n]['random_tokens'][probe+'_'+cond]['mean'] for n in head_ids]
                ses=[evidence[n]['random_tokens'][probe+'_'+cond]['standard_error']*1.96 for n in head_ids]
                ax.bar(np.arange(len(head_ids))+(j-.5)*.36,vals,width=.36,yerr=ses,label=cond)
            ax.set_xticks(range(len(head_ids)),[f'{l}.{h}' for l,h in head_ids],rotation=45)
            ax.set_title(probe.capitalize()+' attention probe'); ax.set_ylabel('Attention probability'); ax.legend()
        copy_heads=[(20,10),(20,11),(21,6)]
        for j,(field,label) in enumerate([('positive_top5_copy_rate','Positive OV'),('negative_top5_copy_rate','Negative OV')]):
            axes[2].bar(np.arange(3)+(j-.5)*.36,[copy[n][field]*100 for n in copy_heads],width=.36,label=label)
        axes[2].set_xticks(range(3),[f'{l}.{h}' for l,h in copy_heads]); axes[2].set_title('OV copying capability'); axes[2].set_ylabel('90 names: top-5 copy (%)'); axes[2].legend()
        finish(fig,pdf,'04_functions','Independent functional probes',
            'Random-token probes: 64 repeated and 64 unique sequences, adapted to BLOOM. 95% normal error bars. Selected published heads shown for comparison only. OV probes test capability; attention patterns alone do not identify a causal circuit.')
        fig,axes=plt.subplots(1,3,figsize=(14,6))
        conditions=['original','name_swap','order_swap','both_swap']
        for ax,field,title in zip(axes[:2],['logit_diff','readout_IO_minus_S1_attention'],['Original-target margin','Readout IO − S1 attention']):
            vals=[counter[c][field]['mean'] for c in conditions]
            ses=[1.96*counter[c][field]['standard_error'] for c in conditions]
            ax.bar(range(4),vals,yerr=ses,color=['#334155','#0d9488','#f59e0b','#8b5cf6'])
            ax.set_xticks(range(4),['Original','Name swap','Order swap','Both'],rotation=25,ha='right'); ax.set_title(title); ax.axhline(0,color='gray',lw=.7)
        observed={(e['layer'],e['head']):e for e in mechanisms['functional_confirmation']}
        for head_id,color in [((20,10),'#0d9488'),((21,6),'#8b5cf6')]:
            s=observed[head_id]['copying_scatter']
            axes[2].scatter(s['io_attention'],s['io_projection'],s=9,alpha=.4,c=color,label=f'{head_id[0]}.{head_id[1]} (r={s["io_pearson"]:.2f})')
        axes[2].set_xlabel('END attention to IO'); axes[2].set_ylabel('IO readout contribution'); axes[2].set_title('Held-out functional confirmation'); axes[2].legend()
        finish(fig,pdf,'05_counterfactuals','Causal sensitivity to names and order',
            '200 development prompts. Joint END-output replacement in subject-focused heads, scored on original IO/S targets. BLOOM uses ALiBi: this tests token/order dependence, not learned-position embedding copying. Scatter correlation is not causal proof.')
        gaps=knockouts['incompleteness']
        fig,ax=plt.subplots(figsize=(13,6))
        knames=list(gaps)
        vals=[gaps[n]['all']['paired_delta']['mean'] for n in knames]
        errors=np.asarray([[vals[i]-knockout_ci[n][0],knockout_ci[n][1]-vals[i]] for i,n in enumerate(knames)]).T
        ax.bar(range(len(knames)),vals,yerr=np.maximum(errors,0),color='#64748b'); ax.axhline(0,color='black',lw=.7)
        ax.set_xticks(range(len(knames)),knames,rotation=45,ha='right',fontsize=8)
        ax.set_ylabel('Mean margin gap: (C\\K) − (M\\K)')
        finish(fig,pdf,'06_completeness','Matched-knockout completeness stress test',
            f'{len(knockouts["evaluation"]["rows"]["full"])} {knockouts["split"]} prompts. Same knockout in circuit C and full model M. Nonzero gaps expose unmodeled compensation. Finite group/random tests do not prove completeness or conditional minimality. Error bars: 95% template-stratified context-cluster bootstrap.')
        fig,ax=plt.subplots(figsize=(14,8))
        positions={}
        role_x={'S1+1':0,'S2':1,'END':2}
        layer_role_counts=Counter((n[0],n[2]) for n in sparse)
        used=Counter()
        for node in sorted(sparse):
            l,h,role=node
            index=used[(l,role)]
            used[(l,role)]+=1
            positions[node]=(role_x[role]+(index-(layer_role_counts[(l,role)]-1)/2)*.13,l)
        from matplotlib.patches import FancyArrowPatch
        selected_edges=[]
        for route in sorted({e['route'] for e in supported_edges}):
            selected_edges.extend(sorted([e for e in supported_edges if e['route']==route],
                key=lambda e:abs(e['delta']),reverse=True)[:8])
        component_colors={'q':'#8b5cf6','k':'#0d9488','v':'#f59e0b'}
        for e in selected_edges:
            a,b=positions[tuple(e['sender'])],positions[tuple(e['receiver_output'])]
            arrow=FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,shrinkA=8,shrinkB=8,
                color=component_colors[e['component']],lw=.6+min(abs(e['delta']),1)*1.3,alpha=.45,
                connectionstyle='arc3,rad=.04')
            ax.add_patch(arrow)
        for n,(x,y) in positions.items():
            ax.scatter(x,y,s=90,c='#0d9488' if n in paper else '#64748b',zorder=3)
            ax.text(x+.035,y+.12,f'{n[0]}.{n[1]}',fontsize=8,zorder=4)
        from matplotlib.lines import Line2D
        handles=[Line2D([0],[0],color=c,lw=2,label=f'{component.upper()} receiver input') for component,c in component_colors.items()]
        handles.extend([Line2D([0],[0],marker='o',ls='',color='#0d9488',label='Published head at matching output position'),
                        Line2D([0],[0],marker='o',ls='',color='#64748b',label='Additional retained node')])
        ax.legend(handles=handles,loc='upper left',fontsize=8)
        ax.set_xlim(-.5,2.8); ax.set_ylim(4.8,24.8); ax.set_ylabel('Layer')
        ax.set_xticks([0,1,2],['Retained output at S1+1','Retained output at S2','Retained output at END'])
        finish(fig,pdf,'07_measured_paths','Measured receiver-input paths in the compact mask',
            f'Shown: {len(selected_edges)} of {len(supported_edges)} supported endpoint paths (up to eight strongest per receiver route), confirmed on 200 separate prompts. Color: Q/K/V input. Induction K is patched at S1+1; subject K/V at S2; Q at END. Output masks differ from input positions. This is a partial graph, not a completeness claim.')
    print(f'Wrote {root / "MEETING_BRIEF.md"}, reproduction report, metrics and seven-page figure PDF')
    print(table())


if __name__=='__main__':
    main()
