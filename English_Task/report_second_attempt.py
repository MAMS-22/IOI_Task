#!/usr/bin/env python3
"""Meeting addendum for the frozen vocabulary-aware whole-head experiment."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'english_results/second_attempt')
    args=parser.parse_args()
    root=args.root
    sys.path.insert(0,'/tmp/ioi-plot-deps')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    load=lambda name:json.loads((root/(name+'.json')).read_text())
    manifest,prune,result=(load(n) for n in ('manifest','prune','evaluate'))
    task_root=root.parent/'second_attempt_task'
    task=json.loads((task_root/'prune.json').read_text())
    assert result['task_frozen_mask_sha256']==hashlib.sha256((task_root/'prune.json').read_bytes()).hexdigest()
    assert task['source_sha256']==hashlib.sha256((task_root/'prune_source.py').read_bytes()).hexdigest()
    for artifact in (prune,task):
        assert max(v['mae'] for group in artifact['initial'].values() for v in group.values())<5e-4
    assert prune['status']==result['status']=='complete'
    assert result['frozen_mask_sha256']==hashlib.sha256((root/'prune.json').read_bytes()).hexdigest()
    assert prune['source_sha256']==hashlib.sha256((root/'prune_source.py').read_bytes()).hexdigest()
    assert result['source_sha256']==hashlib.sha256((root/'evaluate_source.py').read_bytes()).hexdigest()
    old_root=root.parent/'meeting_reproduction'
    old_manifest=json.loads((old_root/'manifest.json').read_text())
    old_prompts={ex['prompt'] for split in ('discovery','confirmation','test') for ex in old_manifest[split]}
    for directory in ('english_circuit_seed42','pipeline_verification','circuit_smoke'):
        path=root.parent/directory/'results.json'
        if path.exists():
            artifact=json.loads(path.read_text())
            for split in ('baseline_examples','discovery_examples','validation_examples'):
                old_prompts.update(ex['prompt'] for ex in artifact.get(split,[]))
    dev_prompts={ex['prompt'] for ex in manifest['development']}
    test_prompts={ex['prompt'] for ex in manifest['test']}
    assert not old_prompts.intersection(dev_prompts|test_prompts)
    assert not dev_prompts.intersection(test_prompts)
    evaluation=result['evaluation']
    summary,rows,fidelity=(evaluation[k] for k in ('summaries','rows','fidelity'))
    ids=[r['example_id'] for r in rows['full']]
    assert all([r['example_id'] for r in group]==ids for group in rows.values())
    contexts={}
    templates={}
    for i,row in enumerate(rows['full']):
        c=row['example_id'].split('_')[0]
        contexts.setdefault(c,[]).append(i)
        templates[c]=row['template_id']
    assert all(len(x)==2 for x in contexts.values())
    rng=np.random.default_rng(202610146)
    pieces=[]
    for t in sorted(set(templates.values())):
        indices=np.asarray([contexts[c] for c in sorted(contexts) if templates[c]==t])
        pieces.append(indices[rng.integers(len(indices),size=(2000,len(indices)))].reshape(2000,-1))
    draw=np.concatenate(pieces,axis=1)
    full=np.asarray([r['logit_diff'] for r in rows['full']])
    names=['full','previous_compact','previous_whole_heads','paper_positions','paper_whole_heads','task_whole_heads','revised_whole_heads']
    labels=['Full model','Previous compact: 33 positions','Previous compact: 33 whole heads','Paper 38: inferred positions','Paper 38: whole heads','Conditional task circuit','Vocabulary-preserving circuit']
    plot_labels=['Full model','Previous 33\npositions','Previous 33\nwhole heads','Paper 38\npositions','Paper 38\nwhole heads','Conditional\ntask circuit','Vocabulary\ncircuit']
    intervals={}
    per_example={}
    for name in names:
        values=np.asarray([r['logit_diff'] for r in rows[name]])
        agreement=np.asarray([a['top_token_id']==b['top_token_id'] for a,b in zip(rows[name],rows['full'])])
        per_example[name]=dict(margin_mae=float(np.abs(values-full).mean()),top1_agreement=float(agreement.mean()))
        intervals[name]=dict(ratio=list(np.quantile(values[draw].mean(1)/full[draw].mean(1),[.025,.975])),
            top1=list(np.quantile(np.asarray([r['zero_rank'] for r in rows[name]])[draw].mean(1),[.025,.975])),
            agreement=list(np.quantile(agreement[draw].mean(1),[.025,.975])))
    pattern_checks={}
    for name in ('task_whole_heads','revised_whole_heads'):
        pattern_checks[name]={}
        for pattern in ('ABBA','BABA'):
            pairs=[(a,b) for a,b in zip(rows[name],rows['full']) if a['pattern']==pattern]
            differences=[a['logit_diff']-b['logit_diff'] for a,b in pairs]
            scale=abs(summary['full'][pattern]['logit_diff']['mean'])
            mean_delta=sum(differences)/len(differences)
            mae=sum(abs(v) for v in differences)/len(differences)
            agreement=sum(a['top_token_id']==b['top_token_id'] for a,b in pairs)/len(pairs)
            passes=abs(mean_delta)<=.05*scale and mae<=.1*scale and summary[name][pattern]['accuracy']>=summary['full'][pattern]['accuracy']-.01
            if name=='revised_whole_heads':
                passes=passes and agreement>=.95 and summary[name][pattern]['zero_rank_rate']>=summary['full'][pattern]['zero_rank_rate']-.02
            pattern_checks[name][pattern]=dict(mean_delta=mean_delta,mae=mae,normalized_mae=mae/scale,
                top1_agreement=agreement,passes_intact_development_targets=passes)
    nheads=len(prune['candidate_nodes'])
    task_count=len(task['candidate_nodes'])
    task_retained={(n['layer'],n['head']) for n in task['candidate_nodes']}
    retained={(n['layer'],n['head']) for n in prune['candidate_nodes']}
    paper_ref=json.loads((Path(__file__).resolve().parent/'paper_bloom_reference.json').read_text())
    overlap={label:dict(total=len(g['heads']),retained=[h for h in g['heads'] if tuple(h) in retained],
        missing=[h for h in g['heads'] if tuple(h) not in retained]) for label,g in paper_ref['groups'].items()}
    matches=sum(len(g['retained']) for g in overlap.values())
    task_matches=sum(tuple(h) in task_retained for g in paper_ref['groups'].values() for h in g['heads'])
    task_overlap={label:dict(total=len(g['heads']),retained=[h for h in g['heads'] if tuple(h) in task_retained],
        missing=[h for h in g['heads'] if tuple(h) not in task_retained]) for label,g in paper_ref['groups'].items()}
    gaps=result['knockout_gaps']
    trained=[n for n in gaps if not n.startswith('unseen')]
    unseen=[n for n in gaps if n.startswith('unseen')]
    worst_trained=max(trained,key=lambda n:abs(gaps[n]['all']['delta']['mean']))
    worst_unseen=max(unseen,key=lambda n:abs(gaps[n]['all']['delta']['mean']))
    worst_pattern=max((abs(v[p]['delta']['mean']),name,p) for name,v in gaps.items() for p in ('ABBA','BABA'))
    task_gaps=result['task_knockout_gaps']
    task_worst_unseen=max(unseen,key=lambda n:abs(task_gaps[n]['all']['delta']['mean']))
    task_worst_trained=max(trained,key=lambda n:abs(task_gaps[n]['all']['delta']['mean']))
    task_worst_pattern=max((abs(v[p]['delta']['mean']),name,p) for name,v in task_gaps.items() for p in ('ABBA','BABA'))
    worst_unseen_mae=max(gaps[n]['all']['mae'] for n in unseen)
    task_worst_unseen_mae=max(task_gaps[n]['all']['mae'] for n in unseen)
    common_max={prefix:max(abs(v[prefix]['all']['delta']['mean']) for v in result['common_control_gaps'].values()) for prefix in ('V','T','P')}
    common_ci={}
    for label in result['common_control_gaps']:
        common_ci[label]={}
        for prefix in ('P','T','V'):
            differences=np.asarray([a['logit_diff']-b['logit_diff'] for a,b in zip(rows[prefix+'_'+label],rows['M_'+label])])
            common_ci[label][prefix]=list(np.quantile(differences[draw].mean(1),[.025,.975]))
    gap_ci={}
    task_gap_ci={}
    for name in gaps:
        a=rows['C_without_'+name]; b=rows['M_without_'+name]
        delta=np.asarray([x['logit_diff']-y['logit_diff'] for x,y in zip(a,b)])
        gap_ci[name]=list(np.quantile(delta[draw].mean(1),[.025,.975]))
        delta_task=np.asarray([x['logit_diff']-y['logit_diff'] for x,y in zip(rows['T_without_'+name],b)])
        task_gap_ci[name]=list(np.quantile(delta_task[draw].mean(1),[.025,.975]))
    pct=lambda x:f'{100*x:.1f}%'
    table=['| Condition | Retention | IO > S | Vocabulary top-1 | Margin/full | Top-token agreement | Margin MAE |',
           '|---|---|---:|---:|---:|---:|---:|']
    for name,label in zip(names,labels):
        s=summary[name]['all']
        retention='all 384 heads' if name=='full' else (f'{nheads} whole heads' if name=='revised_whole_heads' else (f'{task_count} whole heads' if name=='task_whole_heads' else ('38 whole heads' if name=='paper_whole_heads' else ('33 head-positions' if name=='previous_compact' else '38 head-positions'))))
        if name=='previous_whole_heads': retention='33 whole heads'
        ratio=1 if name=='full' else fidelity[name]['all']['fraction_of_full_logit_diff']
        table.append(f"| {label} | {retention} | {pct(s['accuracy'])} | {pct(s['zero_rank_rate'])} | {pct(ratio)} | {pct(per_example[name]['top1_agreement'])} | {per_example[name]['margin_mae']:.3f} |")
    table='\n'.join(table)
    metrics=dict(created_utc=datetime.now(timezone.utc).isoformat(),head_count=nheads,task_head_count=task_count,task_paper_matches=task_matches,paper_overlap=overlap,task_paper_overlap=task_overlap,
        matched_paper_heads=matches,intervals=intervals,per_example=per_example,pattern_checks=pattern_checks,knockout_intervals=gap_ci,task_knockout_intervals=task_gap_ci,common_knockout_maxima=common_max,common_knockout_intervals=common_ci,
        bootstrap='2,000 matched-context clusters, stratified by fixed template; no discovery-seed or template-bank uncertainty',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.json') if p.name!='metrics.json'},
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    s=summary['revised_whole_heads']['all']
    ratio=fidelity['revised_whole_heads']['all']['fraction_of_full_logit_diff']
    brief=f'''# Second attempt: preserve model behavior under circuit reduction

On a new **1,000-prompt test (500 paired contexts)**, the frozen **{nheads}-whole-head** circuit achieves **{pct(s['accuracy'])} IO-vs-subject accuracy**, **{pct(s['zero_rank_rate'])} vocabulary top-1**, and **{pct(ratio)} retained mean margin**. Top-token agreement with the full model is **{pct(per_example['revised_whole_heads']['top1_agreement'])}**; per-prompt margin MAE is **{per_example['revised_whole_heads']['margin_mae']:.3f} logits**.

The parallel **conditional task** variant retains **{task_count} whole heads**, with **{pct(summary['task_whole_heads']['all']['accuracy'])} IO-vs-S accuracy**, **{pct(summary['task_whole_heads']['all']['zero_rank_rate'])} vocabulary top-1**, and **{pct(fidelity['task_whole_heads']['all']['fraction_of_full_logit_diff'])} retained margin**. It uses the same conditional margin checks but does not constrain vocabulary predictions. Both masks were frozen before evaluating this shared fresh test.

This is a different and larger circuit representation: a whole head is retained at every token position. It improves the behavior-preservation objective; it does not establish that the paper's sparse 38-head functional circuit has been exactly reproduced. Compare retention units explicitly.

## Common fresh-test comparison

{table}

All rows use the same new test prompts and the same independent 3,000-prompt ABC mean bank. Embeddings and MLPs remain present. The published-reference masks are our evaluations with inferred positional conventions, not faithfulness numbers reported by the authors.

The previous 33-head inventory is evaluated both at its original semantic positions and at all positions. This control separates changing the retention granularity from changing which heads are selected; comparison only against the positional circuit would mix those effects.

## What changed, and why

The first pruning pass optimized a lower bound on mean IO−S margin and binary accuracy. That could discard negative contributions and preserve the two-name answer while changing vocabulary predictions. This attempt penalizes both increases and decreases in margin, measures per-prompt error and top-token agreement, and checks conditional behavior under six functional-group knockouts.

Both searches began with all 384 whole heads. Eighteen anchor heads were selected from prior independently inferred functional families (up to three per family), using routing, copy-readout sign and repeated-token probes. Those anchors were protected and defined the development knockouts; published IDs were not used. Block removal (16/4/1 heads) used the same 200 new development prompts. The vocabulary constraint caused limited early pruning on development, so the task-only variant was added before any fresh test evaluation. Both circuits were frozen before that test; its results did not revise either mask.

Per ordering, the development constraints were: intact absolute mean bias <=5% and mean absolute error <=10% of the full-model mean margin; >=95% top-token agreement; no more than one percentage point IO-vs-S accuracy loss or two points vocabulary top-1 loss. Under each anchor knockout, absolute mean bias <=8% and MAE <=15%. These are empirical tolerances, not a proof of minimality or completeness.

The development targets do not all generalize to the fresh test. In particular, the vocabulary variant's pooled top-token agreement is {pct(per_example['revised_whole_heads']['top1_agreement'])}, below its 95% development requirement, and its vocabulary top-1 loss is {100*(summary['full']['all']['zero_rank_rate']-s['zero_rank_rate']):.1f} percentage points. Per-ordering checks are recorded in `metrics.json`. These deviations are reported without retuning either mask on the final test.

## Completeness evidence

The largest pooled absolute gap for the six development-selected knockout groups on fresh test prompts is **{abs(gaps[worst_trained]['all']['delta']['mean']):.3f} logits** ({worst_trained}). For **ten previously unused random knockout subsets**, it is **{abs(gaps[worst_unseen]['all']['delta']['mean']):.3f} logits** ({worst_unseen}). The largest per-ordering gap across these tests is **{worst_pattern[0]:.3f} logits** ({worst_pattern[1]}, {worst_pattern[2]}).

For the conditional task variant, the largest pooled gap is **{abs(task_gaps[task_worst_trained]['all']['delta']['mean']):.3f}** for anchor groups and **{abs(task_gaps[task_worst_unseen]['all']['delta']['mean']):.3f} logits** for the same ten random subsets. These random subsets were sampled from heads retained by both circuits, so the two new variants receive exactly the same knockouts. They do not cover subsets unique to either circuit.

The task variant's largest per-ordering mean gap is **{task_worst_pattern[0]:.3f} logits** ({task_worst_pattern[1]}, {task_worst_pattern[2]}). Worst per-prompt MAE over the ten unseen random subsets is **{worst_unseen_mae:.3f} logits** for the vocabulary variant and **{task_worst_unseen_mae:.3f}** for the task variant. Mean gaps alone can hide per-prompt errors and ordering cancellation.

A separate controlled comparison uses **ten identical whole-head knockout subsets shared by all three inventories** on these same fresh prompts. Worst pooled absolute gaps: previous 33 heads retained at all positions **{common_max['P']:.3f} logits**; conditional task variant **{common_max['T']:.3f}**; vocabulary variant **{common_max['V']:.3f}**. This comparison controls prompts, knockout subsets and ablation granularity. It covers only subsets of heads shared by the three inventories, so it is a limited completeness diagnostic.

Finite conditional agreement is evidence of robustness, not exhaustive completeness. The first attempt's 3.020-logit worst gap used a different cohort, positional mask and knockout subsets; do not treat a difference between those maxima as a controlled improvement estimate. The common fresh-test intact comparison above is controlled.

## Comparison with the paper

The vocabulary-preserving inventory overlaps **{matches}/38** published heads; the conditional task inventory overlaps **{task_matches}/38**. This is head-identity overlap only: a whole-head retention mask does not prove the published positional role or exact connecting paths. The prior discovery run's functional probes and receiver-input experiments remain the evidence for those mechanisms.

| Published family | Vocabulary / published | Task / published | Missing from vocabulary variant | Missing from task variant |
|---|---:|---:|---|---|
'''
    for label,g in overlap.items():
        missing=', '.join(f'{l}.{h}' for l,h in g['missing']) or 'none'
        t=task_overlap[label]
        task_missing=', '.join(f'{l}.{h}' for l,h in t['missing']) or 'none'
        brief+=f"| {label} | {len(g['retained'])}/{g['total']} | {len(t['retained'])}/{t['total']} | {missing} | {task_missing} |\n"
    brief+='''
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
'''
    (root/'MEETING_ADDENDUM.md').write_text(brief)
    (root/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#334155','#ef4444','#fb923c','#8b5cf6','#a78bfa','#f59e0b','#0d9488']
    def finish(fig,pdf,name,title,caption):
        fig.suptitle(title,fontsize=16)
        fig.text(.035,.025,caption,fontsize=8,wrap=True)
        fig.tight_layout(rect=(0,.08,1,.93))
        pdf.savefig(fig)
        fig.savefig(root/'figures'/f'{name}.png',dpi=180)
        plt.close(fig)
    with PdfPages(root/'second_attempt_figures.pdf') as pdf:
        fig,axes=plt.subplots(1,3,figsize=(14,6))
        for ax,field,title in zip(axes,['zero_rank_rate','ratio','agreement'],['Vocabulary top-1','Mean margin / full','Top-token agreement with full']):
            vals=[100*(summary[n]['all']['zero_rank_rate'] if field=='zero_rank_rate' else
                ((1 if n=='full' else fidelity[n]['all']['fraction_of_full_logit_diff']) if field=='ratio' else per_example[n]['top1_agreement'])) for n in names]
            interval_key='top1' if field=='zero_rank_rate' else field
            errors=np.asarray([[v-100*intervals[n][interval_key][0],100*intervals[n][interval_key][1]-v] for n,v in zip(names,vals)]).T
            ax.bar(range(len(names)),vals,color=colors,yerr=np.maximum(errors,0),capsize=2)
            ax.set_xticks(range(len(names)),plot_labels,rotation=35,ha='right',fontsize=8)
            ax.set_ylabel('%'); ax.set_title(title)
            for i,v in enumerate(vals): ax.text(i,v+1.5,f'{v:.1f}',ha='center',fontsize=8)
        finish(fig,pdf,'01_fidelity',f'Second attempt: {nheads} whole heads, fresh 1,000-prompt test',
            'Whole heads retain all token positions. Positional masks retain selected output positions only. Same prompts and ABC reference bank. 95% matched-context bootstrap intervals; zero observed error can yield degenerate intervals.')
        fig,ax=plt.subplots(figsize=(14,6))
        common_labels=list(result['common_control_gaps'])
        x=np.arange(len(common_labels))
        for offset,prefix,color,label in [(-.25,'P','#ef4444','Previous 33 whole heads'),
                (0,'T','#f59e0b',f'Conditional task: {task_count} heads'),
                (.25,'V','#0d9488',f'Vocabulary objective: {nheads} heads')]:
            values=[result['common_control_gaps'][g][prefix]['all']['delta']['mean'] for g in common_labels]
            errors=np.asarray([[values[i]-common_ci[g][prefix][0],common_ci[g][prefix][1]-values[i]] for i,g in enumerate(common_labels)]).T
            ax.bar(x+offset,values,width=.25,color=color,label=label,yerr=np.maximum(errors,0),capsize=2)
        ax.axhline(0,color='black',lw=.7)
        ax.set_xticks(x,[f'Subset {i+1}' for i in range(len(common_labels))])
        ax.set_ylabel('Mean (C\\K) − (M\\K), logits'); ax.legend()
        finish(fig,pdf,'02_common_knockouts','Controlled comparison: identical knockouts and fresh prompts',
            f'Each subset removes {len(next(iter(result["common_control_subsets"].values())))} whole heads shared by all three masks. Same reference means and full-model knockout comparator. 95% matched-context bootstrap intervals. Covers shared subsets only; it is not an exhaustive completeness test.')
        fig,axes=plt.subplots(1,2,figsize=(14,6))
        x=np.arange(len(gaps)); glabels=list(gaps)
        vals=[gaps[g]['all']['delta']['mean'] for g in glabels]
        errs=np.asarray([[vals[i]-gap_ci[g][0],gap_ci[g][1]-vals[i]] for i,g in enumerate(glabels)]).T
        axes[0].bar(x-.2,vals,width=.4,yerr=np.maximum(errs,0),color='#0d9488',label='Vocabulary objective')
        task_vals=[task_gaps[g]['all']['delta']['mean'] for g in glabels]
        task_errs=np.asarray([[task_vals[i]-task_gap_ci[g][0],task_gap_ci[g][1]-task_vals[i]] for i,g in enumerate(glabels)]).T
        axes[0].bar(x+.2,task_vals,width=.4,yerr=np.maximum(task_errs,0),color='#f59e0b',label='Task objective')
        axes[0].legend(fontsize=8)
        axes[0].set_xticks(x,glabels,rotation=50,ha='right',fontsize=7)
        axes[0].axhline(0,color='black',lw=.6); axes[0].set_ylabel('Mean (C\\K) − (M\\K), logits')
        axes[0].set_title('Trained anchor groups and unseen random subsets')
        random=[fidelity[f'matched_random_{i}']['all']['fraction_of_full_logit_diff']*100 for i in range(5)]
        axes[1].bar(np.arange(5)-.18,random,width=.36,color='#94a3b8',label='Random: vocabulary head count')
        axes[1].bar(np.arange(5)+.18,[fidelity[f'task_matched_random_{i}']['all']['fraction_of_full_logit_diff']*100 for i in range(5)],width=.36,color='#fcd34d',label='Random: task head count')
        axes[1].axhline(ratio*100,color='#0d9488',label='Vocabulary circuit',lw=2)
        axes[1].axhline(fidelity['task_whole_heads']['all']['fraction_of_full_logit_diff']*100,color='#f59e0b',label='Task circuit',lw=2,ls='--')
        axes[1].set_xticks(range(5),[f'Random {i+1}' for i in range(5)]); axes[1].set_ylabel('Mean margin / full (%)'); axes[1].legend()
        finish(fig,pdf,'02_conditional','Conditional fidelity and matched random controls',
            'Anchor knockout groups were used during development; random subsets were not. Both variants receive identical knockouts sampled from their shared heads. Fresh test prompts throughout. Random masks match retained head counts at each layer. Finite tests do not prove completeness.')
        fig,axes=plt.subplots(1,2,figsize=(13,6))
        matrix=np.zeros((24,16))
        for l,h in retained: matrix[l,h]=1
        axes[0].imshow(matrix,origin='lower',aspect='auto',cmap='Greens',vmin=0,vmax=1)
        for g in paper_ref['groups'].values():
            for l,h in g['heads']: axes[0].plot(h,l,'s',mfc='none',mec='#8b5cf6',ms=8)
        axes[0].set_xlabel('Head'); axes[0].set_ylabel('Layer'); axes[0].set_title(f'Vocabulary mask: {nheads}/384 heads; {matches}/38 paper heads')
        gnames=list(overlap)
        axes[1].bar(np.arange(len(gnames))-.25,[overlap[g]['total'] for g in gnames],width=.25,label='Paper inventory',color='#8b5cf6')
        axes[1].bar(np.arange(len(gnames)),[len(overlap[g]['retained']) for g in gnames],width=.25,label='Vocabulary overlap',color='#0d9488')
        axes[1].bar(np.arange(len(gnames))+.25,[len(task_overlap[g]['retained']) for g in gnames],width=.25,label='Task overlap',color='#f59e0b')
        axes[1].set_xticks(range(len(gnames)),gnames,rotation=25,ha='right'); axes[1].set_ylabel('Heads'); axes[1].legend()
        finish(fig,pdf,'03_inventory','Behavioral fidelity has a circuit-size cost',
            'Green cells: retained whole heads. Purple outlines: published heads, used only for evaluation. Head identity overlap does not establish the same functional role or graph. Anchors derive from the prior independent functional probes.')
    print(table)
    print(f'Worst unseen pooled knockout gap: {abs(gaps[worst_unseen]["all"]["delta"]["mean"]):.6f}')
    print(f'Saved {root / "MEETING_ADDENDUM.md"}')


if __name__=='__main__':
    main()
