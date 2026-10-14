#!/usr/bin/env python3
"""Second attempt: symmetric, conditional, vocabulary-aware whole-head pruning.

Published head IDs are evaluation-only. The old semantic-position circuit and
its final test remain untouched. This produces a different ablation granularity.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from transformer_lens import TransformerBridge
from bloom_ioi.data import build_examples,batches
from bloom_ioi.experiment import score_rows,baseline_summary
from bloom_ioi.interventions import Node,token_tensor,circuit_hooks,stats
from bloom_ioi.reproduction import independent_means,means_for_batch,evaluate_circuits


def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['prune','evaluate'],required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parent/'english_results/meeting_reproduction')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'english_results/second_attempt')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--objective',choices=['vocabulary','task'],default='vocabulary')
    args=parser.parse_args()
    root=args.output_dir
    root.mkdir(parents=True,exist_ok=True)
    destination=root/(args.phase+'.json')
    if destination.exists():
        raise SystemExit(f'Checkpoint exists; preserve it: {destination}')
    (root/(args.phase+'_source.py')).write_text(Path(__file__).read_text())
    started=time.monotonic()
    metadata=dict(status='running',phase=args.phase,objective=args.objective,started_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def progress(message):
        print(message,flush=True)
        save(root/(args.phase+'_status.json'),dict(metadata,last_progress=message,elapsed_seconds=time.monotonic()-started))
    previous=json.loads((args.source/'manifest.json').read_text())
    assembled=json.loads((args.source/'assembled.json').read_text())
    functions=json.loads((args.source/'functions.json').read_text())
    old_prune=json.loads((args.source/'prune.json').read_text())
    model=TransformerBridge.boot_transformers(previous['protocol']['model'],device=args.device,dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    seed=202610145
    manifest_path=root/'manifest.json'
    if not manifest_path.exists():
        excluded={e['prompt'] for split in ('discovery','confirmation','test','reference_examples') for e in previous[split]}
        development=build_examples(model.tokenizer,100,seed,exclude_prompts=excluded,vocabulary=previous['vocabulary'])
        excluded.update(e['prompt'] for e in development)
        test=build_examples(model.tokenizer,500,seed+1,exclude_prompts=excluded,vocabulary=previous['vocabulary'])
        manifest=dict(development=development,test=test,seed=seed,reference_policy='Same independently generated 3,000 ABC bank as the first attempt',
            prior_manifest_sha256=hashlib.sha256((args.source/'manifest.json').read_bytes()).hexdigest(),
            scope='New development and fresh final test; whole-head rather than positional ablation; conditional symmetric pruning')
        save(manifest_path,manifest)
    else:
        manifest=json.loads(manifest_path.read_text())
    bank,bank_metadata=independent_means(model,previous['vocabulary'],100,previous['protocol']['seed']+3,
        args.batch_size,progress,examples=previous['reference_examples'])
    if args.phase=='prune':
        evidence={(e['layer'],e['head']):e for e in functions['evidence']}
        labels=sorted({r['proposed_role'] for r in assembled['proposed_roles']} - {'causal_support'})
        anchors={}
        for label in labels:
            eligible={(r['node']['layer'],r['node']['head']) for r in assembled['proposed_roles'] if r['proposed_role']==label}
            def strength(head):
                e=evidence[head]
                if label.startswith('previous'):
                    return e['random_tokens']['previous_unique']['mean']
                if label.startswith(('duplicate','induction')):
                    probe=label.split('_')[0]
                    return e['random_tokens'][probe+'_repeated']['mean']-e['random_tokens'][probe+'_unique']['mean']
                if label.startswith('subject'):
                    return e['ioi']['attention_END_to_S2']['mean']
                return abs(e['ioi']['end_logit_diff_attribution']['mean'])
            anchors[label]=set(sorted(eligible,key=lambda h:(-strength(h),h))[:3])
        protected=set().union(*anchors.values())
        groups={'intact':set(),**anchors}
        cached=[]
        reference={label:[] for label in groups}
        with torch.inference_mode():
            for batch in batches(manifest['development'],args.batch_size):
                tokens=token_tensor(batch,'clean_ids',args.device)
                donor=means_for_batch(bank,batch,args.device)
                cached.append((batch,tokens,donor))
                for label,heads in groups.items():
                    hooks=circuit_hooks(model,donor,batch,[Node(l,h,'ALL') for l,h in sorted(heads)],'remove')
                    reference[label].extend(score_rows(model.run_with_hooks(tokens,fwd_hooks=hooks),batch))
        full=baseline_summary(reference['intact'])
        def measure(heads,labels_to_run=('intact',)):
            results={}
            with torch.inference_mode():
                for label in labels_to_run:
                    kept=[Node(l,h,'ALL') for l,h in sorted(heads-groups[label])]
                    rows=[]
                    for batch,tokens,donor in cached:
                        logits=model.run_with_hooks(tokens,fwd_hooks=circuit_hooks(model,donor,batch,kept,'keep'))
                        rows.extend(score_rows(logits,batch))
                    measurements={}
                    for pattern in ('all','ABBA','BABA'):
                        pairs=[(a,b) for a,b in zip(rows,reference[label]) if pattern=='all' or a['pattern']==pattern]
                        delta=[a['logit_diff']-b['logit_diff'] for a,b in pairs]
                        measurements[pattern]=dict(mean_delta=sum(delta)/len(delta),mae=sum(map(abs,delta))/len(delta),
                            top1_agreement=sum(a['top_token_id']==b['top_token_id'] for a,b in pairs)/len(pairs),
                            accuracy=sum(a['correct'] for a,b in pairs)/len(pairs),
                            top1=sum(a['zero_rank'] for a,b in pairs)/len(pairs))
                    results[label]=measurements
            return results
        def acceptable(measurements):
            for label,result in measurements.items():
                for pattern in ('ABBA','BABA'):
                    value=result[pattern]
                    scale=abs(full[pattern]['logit_diff']['mean'])
                    if abs(value['mean_delta'])>(.05 if label=='intact' else .08)*scale:
                        return False
                    if value['mae']>(.10 if label=='intact' else .15)*scale:
                        return False
                    if label=='intact' and value['accuracy']<full[pattern]['accuracy']-.01:
                        return False
                    if label=='intact' and args.objective=='vocabulary' and (value['top1_agreement']<.95
                        or value['top1']<full[pattern]['zero_rank_rate']-.02):
                        return False
            return True
        heads={(l,h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)}
        initial=measure(heads,groups)
        assert acceptable(initial),'Full attention identity failed symmetric target checks'
        ranking=[]
        for i,head in enumerate(sorted(heads-protected),1):
            result=measure(heads-{head})
            score=max(result['intact'][pat]['mae']/abs(full[pat]['logit_diff']['mean'])+
                (1-result['intact'][pat]['top1_agreement'] if args.objective=='vocabulary' else 0) for pat in ('ABBA','BABA'))
            ranking.append((score,head))
            if i%25==0: progress(f'Whole-head ranking: {i}/{len(heads-protected)}')
        ranking=[h for score,h in sorted(ranking)]
        trace=[]
        current=initial
        for block_size in (16,4,1):
            remaining=[h for h in ranking if h in heads]
            for offset in range(0,len(remaining),block_size):
                removed=set(remaining[offset:offset+block_size])
                trial=measure(heads-removed)
                accepted=acceptable(trial)
                if accepted:
                    trial.update(measure(heads-removed,[label for label in groups if label!='intact']))
                    accepted=acceptable(trial)
                trace.append(dict(block_size=block_size,removed=sorted(removed),accepted=accepted,measurements=trial))
                if accepted:
                    heads.difference_update(removed)
                    current=trial
                if len(trace)%10==0:
                    progress(f'Symmetric conditional pruning: block {block_size}, {len(heads)} heads, {len(trace)} trials')
        metadata.update(candidate_nodes=[Node(l,h,'ALL').as_dict() for l,h in sorted(heads)],
            anchors={label:[list(h) for h in sorted(group)] for label,group in anchors.items()},
            protected_heads=sorted(protected),initial=initial,final=current,full=full,trace=trace,
            policy='All 384 whole heads initially. Up to three independently inferred anchors per functional family protected. Per-pattern intact bias<=5%, MAE<=10% of full margin, IO accuracy loss<=1pp. Vocabulary objective additionally requires top-token agreement>=95% and IO top1 loss<=2pp. Anchor-knockout bias<=8%, MAE<=15%. Thresholds fixed before final test.',
            selection_used_paper_ids=False,selection_used_final_test=False,
            final_test_sha256=hashlib.sha256(json.dumps(manifest['test'],sort_keys=True).encode()).hexdigest())
        progress(f'Frozen conditional whole-head circuit: {len(heads)} heads')
    else:
        frozen=json.loads((root/'prune.json').read_text())
        nodes=[Node(**n) for n in frozen['candidate_nodes']]
        old=[Node(**n) for n in old_prune['candidate_nodes']]
        reference_file=json.loads((Path(__file__).resolve().parent/'paper_bloom_reference.json').read_text())
        paper=[Node(l,h,g['role']) for g in reference_file['groups'].values() for l,h in g['heads']]
        paper_whole=sorted({Node(n.layer,n.head,'ALL') for n in paper})
        conditions={'previous_compact':old,'revised_whole_heads':nodes,'paper_positions':paper,'paper_whole_heads':paper_whole}
        task_path=root.parent/'second_attempt_task/prune.json'
        task=None
        if task_path.exists() and task_path.parent!=root:
            task=json.loads(task_path.read_text())
            assert task['status']=='complete'
            assert task['final_test_sha256']==frozen['final_test_sha256']
            task_nodes=[Node(**n) for n in task['candidate_nodes']]
            conditions['task_whole_heads']=task_nodes
            metadata['task_frozen_mask_sha256']=hashlib.sha256(task_path.read_bytes()).hexdigest()
        removals={'revised_removed':nodes}
        rng=random.Random(seed+2)
        for trial in range(5):
            random_nodes=[]
            for layer in range(model.cfg.n_layers):
                number=sum(n.layer==layer for n in nodes)
                random_nodes.extend(Node(layer,h,'ALL') for h in rng.sample(range(model.cfg.n_heads),number))
            conditions[f'matched_random_{trial}']=random_nodes
        subsets={label:[Node(l,h,'ALL') for l,h in group] for label,group in frozen['anchors'].items()}
        shared_nodes=sorted(set(nodes).intersection(task_nodes)) if task else nodes
        for i in range(10):
            subsets[f'unseen_random_{i}']=rng.sample(shared_nodes,max(1,len(shared_nodes)//3))
        for label,removed in subsets.items():
            conditions['C_without_'+label]=sorted(set(nodes)-set(removed))
            if task:
                assert set(removed).issubset(task_nodes)
                conditions['T_without_'+label]=sorted(set(task_nodes)-set(removed))
            removals['M_without_'+label]=removed
        evaluation=evaluate_circuits(model,manifest['test'],bank,conditions,args.batch_size,progress,removals)
        gaps={}
        for label in subsets:
            paired=list(zip(evaluation['rows']['C_without_'+label],evaluation['rows']['M_without_'+label]))
            gaps[label]={}
            for pattern in ('all','ABBA','BABA'):
                pairs=[(a,b) for a,b in paired if pattern=='all' or a['pattern']==pattern]
                delta=[a['logit_diff']-b['logit_diff'] for a,b in pairs]
                gaps[label][pattern]=dict(delta=stats(delta),mae=sum(map(abs,delta))/len(delta))
        metadata.update(evaluation=evaluation,knockout_gaps=gaps,subsets={k:[n.as_dict() for n in v] for k,v in subsets.items()},
            conditions={k:[n.as_dict() for n in v] for k,v in conditions.items()},
            frozen_mask_sha256=hashlib.sha256((root/'prune.json').read_bytes()).hexdigest(),
            scope='Fresh test. Whole-head and positional masks clearly distinguished. Anchor subsets trained on new development; random knockout subsets never used during pruning. Finite stress tests are not completeness proof.')
        if task:
            task_gaps={}
            for label in subsets:
                paired=list(zip(evaluation['rows']['T_without_'+label],evaluation['rows']['M_without_'+label]))
                task_gaps[label]={}
                for pattern in ('all','ABBA','BABA'):
                    pairs=[(a,b) for a,b in paired if pattern=='all' or a['pattern']==pattern]
                    delta=[a['logit_diff']-b['logit_diff'] for a,b in pairs]
                    task_gaps[label][pattern]=dict(delta=stats(delta),mae=sum(map(abs,delta))/len(delta))
            metadata['task_knockout_gaps']=task_gaps
    metadata.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
    save(destination,metadata)
    save(root/(args.phase+'_status.json'),metadata)
    print(f'Saved {destination}',flush=True)


if __name__=='__main__':
    main()
