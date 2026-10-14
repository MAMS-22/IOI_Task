#!/usr/bin/env python3
"""Compare discovered nodes against Figure 2 and evaluate common ablation controls.

Reference nodes are an oracle diagnostic, not a blind discovery result. Their
semantic output positions are inferred from Wang/Easy-Transformer conventions;
the supplied BLOOM paper does not provide an executable retention mask.
"""
import argparse
import json
import os
from pathlib import Path
import random

from bloom_ioi.data import name_pool, abc_control, batches
from bloom_ioi.interventions import Node, token_tensor, circuit_hooks
from bloom_ioi.experiment import score_rows, baseline_summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path)
    p.add_argument('--reference',type=Path,default=Path(__file__).resolve().parent/'paper_bloom_reference.json')
    p.add_argument('--device',default='cuda')
    p.add_argument('--donors',type=int,default=8)
    p.add_argument('--seed',type=int,default=44)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--output',type=Path)
    p.add_argument('--skip-model',action='store_true',help='Only compute set overlap and probe measurements')
    args=p.parse_args()
    if args.donors<1 or args.batch_size<1:
        p.error('donors and batch-size must be positive')
    run=json.loads(args.results.read_text())
    reference=json.loads(args.reference.read_text())
    found={Node(**n) for n in run['candidate_nodes']}
    expected={Node(l,h,g['role']) for g in reference['groups'].values() for l,h in g['heads']}
    evidence={(e['layer'],e['head']):e for e in run.get('functions',run.get('selection_evidence',[]))}
    group_comparison={}
    for label,group in reference['groups'].items():
        nodes={Node(l,h,group['role']) for l,h in group['heads']}
        group_comparison[label]=dict(expected=[n.as_dict() for n in sorted(nodes)],
                                     recovered=[n.as_dict() for n in sorted(nodes&found)],
                                     missing=[n.as_dict() for n in sorted(nodes-found)])
    comparison=dict(reference=reference,discovery_config=run['config'],
        group_comparison=group_comparison,
        counts=dict(paper_nodes=len(expected),discovered_nodes=len(found),shared_nodes=len(found&expected),
                    missing_paper_nodes=len(expected-found),additional_nodes=len(found-expected),union_nodes=len(found|expected)),
        additional_nodes=[n.as_dict() for n in sorted(found-expected)],
        paper_head_functional_measurements=[evidence[(n.layer,n.head)] for n in sorted(expected) if (n.layer,n.head) in evidence])
    if not args.skip_model:
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from transformer_lens import TransformerBridge
        model=TransformerBridge.boot_transformers(run['config']['model'],device=args.device,dtype=torch.float32)
        model.eval()
        rng=random.Random(args.seed)
        names=name_pool(model.tokenizer)
        examples=run['validation_examples']
        conditions={'current_nodes':sorted(found),'paper_role_nodes':sorted(expected),
                    'union_role_nodes':sorted(found|expected),
                    'paper_whole_heads':[Node(n.layer,n.head,'ALL') for n in sorted(expected)]}
        rows={label:[] for label in ('full','empty_attention',*conditions)}
        with torch.inference_mode():
            for i,batch in enumerate(batches(examples,args.batch_size),1):
                tokens=token_tensor(batch,'clean_ids',args.device)
                mean_cache={}
                for _ in range(args.donors):
                    ids=[abc_control(ex,model.tokenizer,rng,names) for ex in batch]
                    logits,cache=model.run_with_cache(torch.tensor(ids,device=args.device),
                        names_filter=lambda n:n.endswith('attn.hook_z'))
                    for name,value in cache.items():
                        if name not in mean_cache:
                            mean_cache[name]=value.clone()/args.donors
                        else:
                            mean_cache[name].add_(value/args.donors)
                    del logits,cache
                rows['full'].extend(score_rows(model(tokens),batch))
                for label,nodes in {'empty_attention':[],**conditions}.items():
                    logits=model.run_with_hooks(tokens,fwd_hooks=circuit_hooks(model,mean_cache,batch,nodes))
                    rows[label].extend(score_rows(logits,batch))
                    del logits
                del mean_cache
                print(f'Common-control evaluation: batch {i}',flush=True)
        summaries={label:baseline_summary(values) for label,values in rows.items()}
        fidelity={}
        for label in conditions:
            per_pattern={}
            for pattern in summaries['full']:
                full,empty,retained=(summaries[k][pattern]['logit_diff']['mean'] for k in ('full','empty_attention',label))
                per_pattern[pattern]=dict(fraction_of_full_logit_diff=retained/full if abs(full)>1e-6 else None,
                    normalized_attention_recovery=(retained-empty)/(full-empty) if abs(full-empty)>1e-6 else None)
            fidelity[label]=per_pattern
        comparison['controlled_evaluation']=dict(seed=args.seed,donors=args.donors,summaries=summaries,fidelity=fidelity,rows=rows,
            scope=f'Same {len(examples)} held-out contexts and identical mean-ABC donor activations across all conditions. Embeddings and MLPs retained. Whole-head retention is a diagnostic expansion, not a paper-verified positional circuit mask.')
        print(json.dumps({k:v['all'] for k,v in summaries.items()},indent=2),flush=True)
        print(json.dumps(fidelity,indent=2),flush=True)
    output=args.output or args.results.parent/'paper_comparison.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(comparison,indent=2,allow_nan=False)+'\n')
    # A --circuit compatible array for an independent later validation run.
    (output.parent/'paper_reference_nodes.json').write_text(json.dumps([n.as_dict() for n in sorted(expected)],indent=2)+'\n')
    print(f'Saved {output.resolve()}',flush=True)


if __name__=='__main__':
    main()
