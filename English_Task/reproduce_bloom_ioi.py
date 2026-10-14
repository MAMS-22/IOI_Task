#!/usr/bin/env python3
"""Deadline-oriented, separately checkpointed BLOOM IOI reproduction.

This runner fixes the earlier fixed top-k search and uses independent,
template-conditioned ABC reference means. Paper head IDs are evaluation-only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'

from bloom_ioi.data import build_examples, ROLES, TEMPLATE_SOURCE
from bloom_ioi.experiment import Experiment
from bloom_ioi.interventions import Node,act_name,token_tensor,patch_head,logit_diffs,stats
from bloom_ioi.data import batches
from bloom_ioi.experiment import attention_at
from bloom_ioi.reproduction import (wang_vocabulary,select_supported,ov_copy_scores,
    independent_means,evaluate_circuits)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def hashes():
    root=Path(__file__).resolve().parent
    paths=[Path(__file__),*sorted((root/'bloom_ioi').glob('*.py'))]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=['prepare','direct','functions','specificity','mechanisms','path','assemble','prune','evaluate','knockouts'])
    p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'english_results/meeting_reproduction')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--model',default='bigscience/bloom-560m')
    p.add_argument('--seed',type=int,default=20261014)
    p.add_argument('--discovery-n',type=int,default=200)
    p.add_argument('--confirmation-n',type=int,default=100)
    p.add_argument('--test-n',type=int,default=500)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--reference-per-template',type=int,default=100)
    p.add_argument('--min-direct-effect',type=float,default=.02)
    p.add_argument('--min-path-effect',type=float,default=.001)
    p.add_argument('--fdr',type=float,default=.05)
    p.add_argument('--target',choices=['positive_output_q','negative_output_q','subject_end_q','subject_end_v','subject_end_k','induction_k'])
    p.add_argument('--evaluate-kind',choices=['reference','discovered'],default='discovered')
    p.add_argument('--split',choices=['discovery','confirmation','test'],default='test')
    p.add_argument('--probe-samples',type=int,default=64)
    p.add_argument('--probe-length',type=int,default=32)
    p.add_argument('--no-bos',action='store_true')
    p.add_argument('--skip-individual-edges',action='store_true',help='Group confirmation only; never changes discovery selection')
    args=p.parse_args()
    if min(args.discovery_n,args.confirmation_n,args.test_n,args.batch_size,args.reference_per_template,args.probe_samples)<1:
        p.error('Counts must be positive')
    if not 0<args.fdr<=1 or min(args.min_direct_effect,args.min_path_effect)<0:
        p.error('Invalid selection parameters')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=args.output_dir/'manifest.json'
    phase_name=args.phase+('_'+args.target if args.phase=='path' and args.target else '')
    if args.phase=='assemble':
        phase_name='assembled'
    if args.phase=='evaluate':
        phase_name+=f'_{args.evaluate_kind}_{args.split}'
    phase_path=args.output_dir/(phase_name+'.json')
    if phase_path.exists():
        existing=json.loads(phase_path.read_text())
        if existing.get('status')=='complete':
            print(f'Already complete: {phase_path}',flush=True)
            return
    started=time.monotonic()
    metadata=dict(phase=phase_name,status='running',started_utc=datetime.now(timezone.utc).isoformat(),
        command_config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},source_hashes=hashes())
    save(args.output_dir/(phase_name+'_status.json'),metadata)
    def progress(message):
        print(message,flush=True)
        save(args.output_dir/(phase_name+'_status.json'),dict(metadata,last_progress=message,elapsed_seconds=time.monotonic()-started))
    if args.phase=='prepare':
        if manifest_path.exists():
            p.error('Manifest already exists; use a new output directory')
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        vocabulary=wang_vocabulary(tokenizer)
        excluded=set()
        old_root=Path(__file__).resolve().parent/'english_results'
        for old in [old_root/'english_circuit_seed42/results.json',old_root/'pipeline_verification/results.json',old_root/'circuit_smoke/results.json']:
            if old.exists():
                artifact=json.loads(old.read_text())
                for key in ('baseline_examples','discovery_examples','validation_examples'):
                    excluded.update(ex['prompt'] for ex in artifact.get(key,[]))
        discovery=build_examples(tokenizer,args.discovery_n,args.seed,prepend_bos=not args.no_bos,
            exclude_prompts=excluded,vocabulary=vocabulary)
        excluded.update(ex['prompt'] for ex in discovery)
        confirmation=build_examples(tokenizer,args.confirmation_n,args.seed+1,prepend_bos=not args.no_bos,
            exclude_prompts=excluded,vocabulary=vocabulary)
        excluded.update(ex['prompt'] for ex in confirmation)
        test=build_examples(tokenizer,args.test_n,args.seed+2,prepend_bos=not args.no_bos,
            exclude_prompts=excluded,vocabulary=vocabulary)
        bank=build_examples(tokenizer,15*args.reference_per_template,args.seed+3,
            prepend_bos=not args.no_bos,vocabulary=vocabulary)
        protocol=dict(model=args.model,seed=args.seed,discovery_n=args.discovery_n,confirmation_n=args.confirmation_n,
            test_n=args.test_n,reference_per_template=args.reference_per_template,bos=not args.no_bos,
            min_direct_effect=args.min_direct_effect,min_path_effect=args.min_path_effect,fdr=args.fdr,
            probe_samples=args.probe_samples,probe_length=args.probe_length)
        manifest=dict(protocol=protocol,vocabulary=vocabulary,discovery=discovery,confirmation=confirmation,test=test,
            reference_examples=bank,template_source=TEMPLATE_SOURCE,
            scope='English BLOOM path-patching method reproduction; adapted Wang lexical pools, not author-verified Zhang corpus',
            selection='Per-pattern paired effect screen with BH-corrected normal-approximation p-values; no family-size cap or hard attention exclusion',
            split_policy='Explicitly disjoint discovery, confirmation and final test prompts; prior diagnostic prompts excluded from all task splits',
            reference_policy='Independent uniform ABC distribution; means conditioned on template/pattern and checked exact semantic alignment',
            source_hashes=hashes(),created_utc=metadata['started_utc'])
        save(manifest_path,manifest)
        metadata['counts']={name:len(manifest[name]) for name in ('discovery','confirmation','test','reference_examples')}
        metadata['vocabulary_counts']={key:len(vocabulary[key]) for key in ('names','places','objects')}
        progress(str(metadata['counts'])+'; vocabulary '+str(metadata['vocabulary_counts']))
    else:
        if not manifest_path.exists():
            p.error('Run --phase prepare first')
        manifest=json.loads(manifest_path.read_text())
        protocol=manifest['protocol']
        import torch
        from transformer_lens import TransformerBridge
        torch.manual_seed(protocol['seed'])
        model=TransformerBridge.boot_transformers(protocol['model'],device=args.device,dtype=torch.float32)
        model.eval()
        exp=Experiment(model,args.batch_size,progress=progress)
        if args.phase=='direct':
            records=exp.scan(manifest['discovery'],'direct',retain_values=True)
            nodes,decisions=select_supported(records,protocol['min_direct_effect'],protocol['fdr'])
            metadata.update(records=records,selected_nodes=[n.as_dict() for n in nodes],selection_decisions=decisions,
                baseline=exp.baseline(manifest['discovery']))
            progress(f'Direct candidates: {len(nodes)}; no paper head IDs used')
        elif args.phase=='functions':
            evidence=exp.functional_evidence(manifest['discovery'],protocol['seed']+100,
                protocol['probe_samples'],protocol['probe_length'])
            copy=ov_copy_scores(model,manifest['vocabulary']['names'])
            metadata.update(evidence=evidence,ov_copy=copy,baseline=exp.baseline(manifest['confirmation']),
                method='Clean IO/S readout separately, random-token probes, and positive/negative OV matrix copying capability')
        elif args.phase=='specificity':
            import copy
            examples=copy.deepcopy(manifest['discovery'])
            for ex in examples:
                ex['corrupt_ids']=ex['clean_ids'].copy()
                ex['corrupt_ids'][ex['positions']['S2']]=ex['io_id']
                ex['corruption']='subject_swap'
            records=exp.scan(examples,'direct',retain_values=True)
            nodes,decisions=select_supported(records,protocol['min_direct_effect'],protocol['fdr'])
            metadata.update(records=records,selected_nodes=[n.as_dict() for n in nodes],selection_decisions=decisions,
                baseline=exp.baseline(examples),
                method='Supplemental direct-path contrast changing only S2 to IO, reversing the correct answer while preserving the first two name tokens. Clean targets retained for scoring. This phase does not change the primary circuit selection.')
        elif args.phase=='mechanisms':
            direct=json.loads((args.output_dir/'direct.json').read_text())
            functions=json.loads((args.output_dir/'functions.json').read_text())
            evidence={(e['layer'],e['head']):e for e in functions['evidence']}
            outputs=[Node(**n) for n in direct['selected_nodes']]
            subject=[]
            readouts=[]
            for node in outputs:
                e=evidence[(node.layer,node.head)]['ioi']
                io,s1,s2=(e[f'attention_END_to_{r}']['mean'] for r in ('IO','S1','S2'))
                if s2>=.1 and s2>max(io,s1):
                    subject.append(node)
                elif io>=.1 and e['end_logit_diff_attribution']['mean']>0:
                    readouts.append(node)
            measurements={condition:[] for condition in ('original','name_swap','order_swap','both_swap')}
            attention_measurements={condition:[] for condition in measurements}
            examples=manifest['confirmation']
            with torch.inference_mode():
                for i,batch in enumerate(batches(examples,args.batch_size),1):
                    tokens=token_tensor(batch,'clean_ids',args.device)
                    for condition in measurements:
                        changed=tokens.clone()
                        for row,ex in enumerate(batch):
                            io,s1,s2=(ex['positions'][r] for r in ('IO','S1','S2'))
                            if condition in ('name_swap','both_swap'):
                                changed[row,io],changed[row,s1],changed[row,s2]=tokens[row,s1],tokens[row,io],tokens[row,io]
                            if condition in ('order_swap','both_swap'):
                                a,b=changed[row,io].clone(),changed[row,s1].clone()
                                changed[row,io],changed[row,s1]=b,a
                        control_logits,donor=model.run_with_cache(changed,names_filter=lambda name:name.endswith('attn.hook_z'))
                        del control_logits
                        hooks=[(act_name(n.layer,'z'),patch_head(donor[act_name(n.layer,'z')],batch,n.head,'END')) for n in subject]
                        patterns={}
                        for name in {act_name(n.layer,'pattern') for n in readouts}:
                            def capture(value,hook,name=name):
                                patterns[name]=value.detach().clone()
                                return value
                            hooks.append((name,capture))
                        logits=model.run_with_hooks(tokens,fwd_hooks=hooks)
                        measurements[condition].extend(logit_diffs(logits,batch).tolist())
                        preference=[]
                        for n in readouts:
                            ptn=patterns[act_name(n.layer,'pattern')]
                            preference.append(attention_at(ptn,batch,n.head,'END','IO')-attention_at(ptn,batch,n.head,'END','S1'))
                        if preference:
                            attention_measurements[condition].extend(torch.stack(preference).mean(0).tolist())
                        del logits,donor,patterns
                    progress(f'Token/order counterfactuals: batch {i}')
            summaries={condition:dict(logit_diff=stats(values),
                delta_logit_vs_original=stats(v-b for v,b in zip(values,measurements['original'])),
                readout_IO_minus_S1_attention=stats(attention_measurements[condition]) if attention_measurements[condition] else None)
                for condition,values in measurements.items()}
            scatter=exp.functional_evidence(examples,protocol['seed']+101,protocol['probe_samples'],protocol['probe_length'],include_scatter=True)
            metadata.update(counterfactual_summaries=summaries,counterfactual_scores=measurements,
                counterfactual_attention=attention_measurements,subject_focused_nodes=[n.as_dict() for n in subject],
                observed_readouts=[n.as_dict() for n in readouts],functional_confirmation=scatter,
                method='Joint END-output activation replacement on original inputs. Global name swap preserves repeated-name position; initial order swap changes that position; both changes both signals. This is a BLOOM-specific token/order diagnostic, not a learned-position-embedding claim.')
        elif args.phase=='path':
            if not args.target:
                p.error('--target is required for path phase')
            direct_path=args.output_dir/'direct.json'
            functions_path=args.output_dir/'functions.json'
            if not direct_path.exists() or not functions_path.exists():
                p.error('Complete direct and functions before paths')
            direct=json.loads(direct_path.read_text())
            functions=json.loads(functions_path.read_text())
            evidence={(e['layer'],e['head']):e for e in functions['evidence']}
            output=[Node(**n) for n in direct['selected_nodes']]
            groups={'positive_output_q':[],'negative_output_q':[],'subject_end_q':[]}
            for node in output:
                e=evidence[(node.layer,node.head)]['ioi']
                io,s1,s2=(e[f'attention_END_to_{r}']['mean'] for r in ('IO','S1','S2'))
                attribution=e['end_logit_diff_attribution']['mean']
                # Families organize receiver tests; every supported END node
                # remains in the circuit regardless of its attention label.
                if s2>max(io,s1):
                    groups['subject_end_q'].append(node)
                elif attribution>=0:
                    groups['positive_output_q'].append(node)
                else:
                    groups['negative_output_q'].append(node)
            if args.target in groups:
                receivers=groups[args.target]
                role,component='END','q'
            elif args.target in ('subject_end_v','subject_end_k'):
                receivers=[Node(n.layer,n.head,'S2') for n in groups['subject_end_q']]
                role='S2'
                component='v' if args.target.endswith('_v') else 'k'
            else:
                upstream=set()
                for target in ('subject_end_v','subject_end_k'):
                    path=args.output_dir/f'path_{target}.json'
                    if not path.exists():
                        p.error('Complete subject_end_v and subject_end_k before induction_k')
                    upstream.update(Node(**n) for n in json.loads(path.read_text())['selected_nodes'])
                receivers=[]
                for node in sorted(upstream):
                    e=evidence[(node.layer,node.head)]
                    rp=e['random_tokens']
                    delta=rp['induction_repeated']['mean']-rp['induction_unique']['mean']
                    se=(rp['induction_repeated']['standard_error']**2+rp['induction_unique']['standard_error']**2)**.5
                    ioi=e['ioi']
                    if delta>max(.05,3*se) and ioi['attention_S2_to_S1+1']['mean']>ioi['attention_S2_to_S1']['mean']:
                        receivers.append(Node(node.layer,node.head,'S1+1'))
                role,component='S1+1','k'
            records=exp.scan(manifest['discovery'],'path',(role,),receivers,component,retain_values=True)
            selected,decisions=select_supported(records,protocol['min_path_effect'],protocol['fdr'])
            confirmation=exp.scan(manifest['confirmation'],'path',(role,),receivers,component,
                sender_nodes=selected,retain_values=True) if selected else []
            confirmed,confirmation_decisions=select_supported(confirmation,protocol['min_path_effect'],protocol['fdr'])
            edges=[] if args.skip_individual_edges else exp.confirm_edges_batched(
                manifest['confirmation'],confirmed,receivers,component)
            metadata.update(records=records,discovery_selected_nodes=[n.as_dict() for n in selected],
                selected_nodes=[n.as_dict() for n in confirmed],selection_decisions=decisions,
                confirmation_records=confirmation,confirmation_decisions=confirmation_decisions,
                receivers=[n.as_dict() for n in receivers],source_role=role,component=component,individual_edges=edges,
                selection_scope='Discovery and independent confirmation only; final-test prompts unused')
            progress(f'{args.target}: {len(selected)} discovery candidates, {len(confirmed)} confirmed senders, {len(edges)} individual edges')
        elif args.phase=='evaluate':
            bank,bank_metadata=independent_means(model,manifest['vocabulary'],protocol['reference_per_template'],
                protocol['seed']+3,args.batch_size,progress,examples=manifest['reference_examples'])
            reference=json.loads((Path(__file__).resolve().parent/'paper_bloom_reference.json').read_text())
            paper=sorted({Node(l,h,group['role']) for group in reference['groups'].values() for l,h in group['heads']})
            old=json.loads((Path(__file__).resolve().parent/'english_results/english_circuit_seed42/results.json').read_text())
            legacy=[Node(**n) for n in old['candidate_nodes']]
            conditions={'paper_reference':paper,'legacy_33_nodes':legacy}
            removals={}
            if args.evaluate_kind=='discovered':
                circuit_path=args.output_dir/'assembled.json'
                if not circuit_path.exists():
                    p.error('Run assemble before evaluating the discovered circuit')
                circuit=json.loads(circuit_path.read_text())
                nodes=[Node(**n) for n in circuit['candidate_nodes']]
                pruned_path=args.output_dir/'prune.json'
                if pruned_path.exists():
                    conditions['confirmed_broad']=nodes
                    pruned=json.loads(pruned_path.read_text())
                    if pruned['status']!='complete':
                        p.error('Pruning must finish before final evaluation')
                    nodes=[Node(**n) for n in pruned['candidate_nodes']]
                    metadata['pruned_circuit_sha256']=hashlib.sha256(pruned_path.read_bytes()).hexdigest()
                conditions['discovered']=nodes
                if circuit.get('expanded_nodes'):
                    conditions['expanded_discovery']=[Node(**n) for n in circuit['expanded_nodes']]
                if circuit.get('functional_core_nodes'):
                    conditions['functional_core']=[Node(**n) for n in circuit['functional_core_nodes']]
                conditions['discovered_whole_heads']=[Node(l,h,'ALL') for l,h in sorted({(n.layer,n.head) for n in nodes})]
                removals['discovered_removed']=nodes
                rng=__import__('random').Random(protocol['seed']+200)
                by_group={}
                for n in nodes:
                    by_group.setdefault((n.layer,n.role),[]).append(n)
                for trial in range(5):
                    random_nodes=[]
                    for (l,role),group in by_group.items():
                        random_nodes.extend(Node(l,h,role) for h in rng.sample(range(model.cfg.n_heads),len(group)))
                    conditions[f'matched_random_{trial}']=random_nodes
                metadata['circuit_sha256']=hashlib.sha256(circuit_path.read_bytes()).hexdigest()
            evaluation=evaluate_circuits(model,manifest[args.split],bank,conditions,args.batch_size,progress,removals)
            metadata.update(evaluation=evaluation,mean_bank_metadata={k:v for k,v in bank_metadata.items() if k!='examples'},
                split=args.split,conditions={k:[n.as_dict() for n in v] for k,v in conditions.items()},
                reference_status='Paper head IDs used only in this benchmark evaluation, with explicitly inferred positional mask')
            progress(json.dumps({k:v['all'] for k,v in evaluation['summaries'].items()}))
        elif args.phase=='assemble':
            direct=json.loads((args.output_dir/'direct.json').read_text())
            functions=json.loads((args.output_dir/'functions.json').read_text())
            names=('positive_output_q','negative_output_q','subject_end_q','subject_end_v','subject_end_k','induction_k')
            path_artifacts=[]
            for name in names:
                path=args.output_dir/f'path_{name}.json'
                if not path.exists():
                    p.error(f'Missing completed path phase {name}')
                artifact=json.loads(path.read_text())
                if artifact['status']!='complete':
                    p.error(f'Path phase not complete: {name}')
                path_artifacts.append(artifact)
            output=[Node(**n) for n in direct['selected_nodes']]
            direct_confirmation=exp.scan(manifest['confirmation'],'direct',sender_nodes=output,retain_values=True)
            confirmed_output,decisions=select_supported(direct_confirmation,protocol['min_path_effect'],protocol['fdr'])
            expanded=set(output)
            confirmed=set(confirmed_output)
            for artifact in path_artifacts:
                expanded.update(Node(**n) for n in artifact['discovery_selected_nodes'])
                confirmed.update(Node(**n) for n in artifact['selected_nodes'])
            evidence={(e['layer'],e['head']):e for e in functions['evidence']}
            core=set()
            proposed=[]
            for node in sorted(confirmed):
                e=evidence[(node.layer,node.head)]
                ioi,rp=e['ioi'],e['random_tokens']
                label='causal_support'
                functional=False
                if node.role=='END':
                    io=ioi['attention_END_to_IO']['mean']
                    s2=ioi['attention_END_to_S2']['mean']
                    dla=ioi['end_logit_diff_attribution']['mean']
                    if s2>=.1 and s2>=io:
                        label='subject_focused_END_candidate'
                        functional=True
                    elif io>=.1 and abs(dla)>=.005:
                        label='positive_name_readout_candidate' if dla>=0 else 'negative_name_readout_candidate'
                        functional=True
                elif node.role=='S2':
                    for role,min_delta,source in [('duplicate',.03,'S1'),('induction',.05,'S1+1')]:
                        delta=rp[role+'_repeated']['mean']-rp[role+'_unique']['mean']
                        se=(rp[role+'_repeated']['standard_error']**2+rp[role+'_unique']['standard_error']**2)**.5
                        other='S1+1' if source=='S1' else 'S1'
                        if delta>max(min_delta,3*se) and ioi['attention_S2_to_'+source]['mean']>ioi['attention_S2_to_'+other]['mean']:
                            label=role+'_candidate'
                            functional=True
                elif node.role=='S1+1' and rp['previous_unique']['mean']>=.5:
                    label='previous_token_candidate'
                    functional=True
                if functional:
                    core.add(node)
                proposed.append(dict(node=node.as_dict(),proposed_role=label,functional_core=functional))
            metadata.update(candidate_nodes=[n.as_dict() for n in sorted(confirmed)],
                expanded_nodes=[n.as_dict() for n in sorted(expanded)],
                functional_core_nodes=[n.as_dict() for n in sorted(core)],
                proposed_roles=proposed,direct_confirmation=direct_confirmation,direct_confirmation_decisions=decisions,
                circuit_policy='Confirmed mask is primary. Expanded and behavior-filtered core masks are reported as predeclared sensitivity analyses, not selected using the final test.',
                selection_used_paper_ids=False,selection_used_final_test=False,
                final_test_manifest_sha256=hashlib.sha256(json.dumps(manifest['test'],sort_keys=True).encode()).hexdigest(),
                individual_edges=[edge for artifact in path_artifacts for edge in artifact['individual_edges']])
            progress(f'Circuit frozen: {len(confirmed)} confirmed, {len(expanded)} expanded, {len(core)} functional-core nodes')
        elif args.phase=='prune':
            from bloom_ioi.interventions import circuit_hooks
            from bloom_ioi.reproduction import means_for_batch
            from bloom_ioi.experiment import score_rows,baseline_summary
            circuit_path=args.output_dir/'assembled.json'
            circuit=json.loads(circuit_path.read_text())
            nodes=[Node(**n) for n in circuit['candidate_nodes']]
            bank,bank_metadata=independent_means(model,manifest['vocabulary'],protocol['reference_per_template'],
                protocol['seed']+3,args.batch_size,progress,examples=manifest['reference_examples'])
            cached=[]
            full_rows=[]
            for batch in batches(manifest['confirmation'],args.batch_size):
                tokens=token_tensor(batch,'clean_ids',args.device)
                donor=means_for_batch(bank,batch,args.device)
                full_rows.extend(score_rows(model(tokens),batch))
                cached.append((batch,tokens,donor))
            full=baseline_summary(full_rows)
            def measure(kept):
                rows=[]
                with torch.inference_mode():
                    for batch,tokens,donor in cached:
                        logits=model.run_with_hooks(tokens,fwd_hooks=circuit_hooks(model,donor,batch,kept,'keep'))
                        rows.extend(score_rows(logits,batch))
                return baseline_summary(rows)
            initial=measure(nodes)
            # Fixed development-only target, separately for both orderings.
            # Preserve initial behavior if its margin is already below 90%.
            targets={pat:dict(margin=min(.9*full[pat]['logit_diff']['mean'],initial[pat]['logit_diff']['mean']),
                              accuracy=min(.99,initial[pat]['accuracy'])) for pat in ('ABBA','BABA')}
            trace=[]
            marginal=[]
            pruning_started=time.monotonic()
            for i,node in enumerate(nodes,1):
                summary=measure([n for n in nodes if n!=node])
                importance=max((initial[pat]['logit_diff']['mean']-summary[pat]['logit_diff']['mean'])/
                    full[pat]['logit_diff']['mean'] for pat in ('ABBA','BABA'))
                marginal.append((importance,node))
                if i%25==0:
                    progress(f'Prune ranking: {i}/{len(nodes)} node knockouts')
            ranked=[n for _,n in sorted(marginal)]
            kept=set(nodes)
            current=initial
            timed_out=False
            for block_size in (16,4,1):
                remaining=[n for n in ranked if n in kept]
                for start in range(0,len(remaining),block_size):
                    if time.monotonic()-pruning_started>1800:
                        timed_out=True
                        break
                    removed=remaining[start:start+block_size]
                    trial=measure(sorted(kept-set(removed)))
                    accepted=all(trial[pat]['logit_diff']['mean']>=targets[pat]['margin'] and
                        trial[pat]['accuracy']>=targets[pat]['accuracy'] for pat in targets)
                    trace.append(dict(block_size=block_size,removed=[n.as_dict() for n in removed],
                        accepted=accepted,summary=trial))
                    if accepted:
                        kept.difference_update(removed)
                        current=trial
                    if len(trace)%10==0:
                        progress(f'Greedy pruning: block {block_size}, {len(kept)} retained, {len(trace)} trials')
                if timed_out:
                    break
            metadata.update(candidate_nodes=[n.as_dict() for n in sorted(kept)],full=full,initial=initial,final=current,
                targets=targets,trace=trace,marginal_ranking=[dict(node=n.as_dict(),importance=v) for v,n in sorted(marginal)],
                budget_reached=timed_out,assembled_sha256=hashlib.sha256(circuit_path.read_bytes()).hexdigest(),
                selection_used_paper_ids=False,selection_used_final_test=False,
                policy='Greedy development-only pruning, blocks 16/4/1, preserves >=90% of full mean margin and >=99% IO-vs-S accuracy per ordering when initial mask meets those targets. Thirty-minute search budget. This is not a proof of minimality.',
                mean_bank_metadata={k:v for k,v in bank_metadata.items() if k!='examples'})
            progress(f'Pruned circuit frozen: {len(kept)} nodes')
        elif args.phase=='knockouts':
            circuit=json.loads((args.output_dir/'assembled.json').read_text())
            nodes=[Node(**n) for n in circuit['candidate_nodes']]
            pruned_path=args.output_dir/'prune.json'
            if pruned_path.exists():
                nodes=[Node(**n) for n in json.loads(pruned_path.read_text())['candidate_nodes']]
            core=[Node(**n) for n in circuit['functional_core_nodes']]
            bank,bank_metadata=independent_means(model,manifest['vocabulary'],protocol['reference_per_template'],
                protocol['seed']+3,args.batch_size,progress,examples=manifest['reference_examples'])
            # Matched knockouts in C and M follow Wang's definition. Finite
            # subsets are a stress test, not an exhaustive completeness proof.
            subsets={'all_END':{n for n in nodes if n.role=='END'},
                     'all_S2':{n for n in nodes if n.role=='S2'},
                     'all_previous_positions':{n for n in nodes if n.role=='S1+1'}}
            for label in sorted({x['proposed_role'] for x in circuit['proposed_roles']}):
                subsets[label]={Node(**x['node']) for x in circuit['proposed_roles'] if x['proposed_role']==label}.intersection(nodes)
            rng=__import__('random').Random(protocol['seed']+300)
            for i in range(10):
                subsets[f'random_subset_{i}']=set(rng.sample(nodes,max(1,len(nodes)//3)))
            conditions={'intact_discovered':nodes,'functional_core':core}
            removals={}
            for label,removed in subsets.items():
                if not removed:
                    continue
                conditions['C_without_'+label]=[n for n in nodes if n not in removed]
                removals['M_without_'+label]=sorted(removed)
            evaluation=evaluate_circuits(model,manifest[args.split],bank,conditions,args.batch_size,progress,removals)
            incompleteness={}
            for label,removed in subsets.items():
                if not removed:
                    continue
                per_pattern={}
                for pattern in ('all','ABBA','BABA'):
                    cr=[r for r in evaluation['rows']['C_without_'+label] if pattern=='all' or r['pattern']==pattern]
                    mr=[r for r in evaluation['rows']['M_without_'+label] if pattern=='all' or r['pattern']==pattern]
                    delta=stats(a['logit_diff']-b['logit_diff'] for a,b in zip(cr,mr))
                    per_pattern[pattern]=dict(paired_delta=delta,absolute_mean_gap=abs(delta['mean']))
                incompleteness[label]=per_pattern
            metadata.update(evaluation=evaluation,incompleteness=incompleteness,
                subsets={k:[n.as_dict() for n in sorted(v)] for k,v in subsets.items()},
                mean_bank_metadata={k:v for k,v in bank_metadata.items() if k!='examples'},
                split=args.split,scope='Matched group and ten random-subset knockouts; no exhaustive completeness or minimality claim')
        else:
            p.error('This phase is being implemented; prepare/direct/functions are available')
    metadata.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
        versions={n:importlib.metadata.version(n) for n in ('torch','transformers','transformer_lens')})
    save(phase_path,metadata)
    save(args.output_dir/(phase_name+'_status.json'),metadata)
    print(f'Saved {phase_path}',flush=True)


if __name__=='__main__':
    main()
