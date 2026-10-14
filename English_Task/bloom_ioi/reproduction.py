"""Independent reference means and hypothesis-light causal candidate selection.

The paper's head IDs are never consulted by the selection functions here.
"""
import ast
from collections import defaultdict
import math
from pathlib import Path
import random
import statistics

import torch

from .data import name_pool, batches, build_examples
from .experiment import baseline_summary, score_rows
from .interventions import Node, act_name, circuit_hooks, token_tensor, stats


def wang_vocabulary(tokenizer):
    path=Path(__file__).resolve().parents[1]/'Easy-Transformer/easy_transformer/ioi_dataset.py'
    tree=ast.parse(path.read_text())
    pools={}
    for label,key in [('NAMES','names'),('PLACES','places'),('OBJECTS','objects')]:
        values=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id==label for t in n.targets))
        pools[key]=[v for v in values if len(tokenizer.encode(' '+v,add_special_tokens=False))==1]
    pools['names']=name_pool(tokenizer,pools['names'])
    pools['source']=str(path)
    pools['filter']='All names, places and objects are one leading-space BLOOM token; name occurrences and answer boundary checked in every prompt.'
    return pools


def select_supported(records, min_effect=.001, fdr=.05, patterns=('ABBA','BABA')):
    """Union of per-pattern two-sided paired tests with BH correction.

    Normal-approximation p-values use the paired effect mean/SE, justified by
    the large prompt count. FDR control is a screen, not proof of functional
    role; independent data and counterfactual tests provide confirmation.
    """
    if not 0<fdr<=1 or min_effect<0:
        raise ValueError('Invalid screening parameters')
    selected=set()
    decisions=[]
    for pattern in patterns:
        tested=[]
        for record in records:
            if pattern not in record['effects']:
                continue
            measurement=record['effects'][pattern]['delta']
            mean,se=measurement['mean'],measurement['standard_error']
            p=math.erfc(abs(mean)/se/math.sqrt(2)) if se>0 else (0. if mean else 1.)
            tested.append((p,Node(**record['node']),mean,se))
        tested.sort(key=lambda v:(v[0],v[1]))
        threshold=0.
        for i,(p,*_) in enumerate(tested,1):
            if p<=fdr*i/max(len(tested),1):
                threshold=p
        for p,node,mean,se in tested:
            accepted=p<=threshold and abs(mean)>=min_effect
            if accepted:
                selected.add(node)
            decisions.append(dict(node=node.as_dict(),pattern=pattern,mean=mean,standard_error=se,
                                  approximate_p=p,bh_cutoff=threshold,accepted=accepted))
    return sorted(selected),decisions


def group_key(example):
    return (example['template_id'],example['pattern'],len(example['clean_ids']),
            *(example['positions'][r] for r in ('IO','S1','S1+1','S2','END')))


@torch.inference_mode()
def independent_means(model, vocabulary, per_template=100, seed=202603, batch_size=16, progress=print,
                      examples=None):
    """Uniform independent ABC reference bank, averaged within template/pattern.

    Single-token lexical slots make absolute grammar positions align within
    each template. The exact positional key is also asserted, not assumed.
    No task evaluation labels or discovery activations enter this mean bank.
    """
    if examples is None:
        examples=build_examples(model.tokenizer,15*per_template,seed,vocabulary=vocabulary)
    device=next(model.parameters()).device
    by_group=defaultdict(list)
    for ex in examples:
        by_group[group_key(ex)].append(ex)
    sums={}
    counts={}
    for i,(key,group) in enumerate(sorted(by_group.items()),1):
        if len(group)!=per_template:
            raise ValueError('Reference lexical slots did not align at the template level')
        sums[key]={}
        count=0
        for batch in batches(group,batch_size):
            logits,cache=model.run_with_cache(token_tensor(batch,'corrupt_ids',device),
                names_filter=lambda name:name.endswith('attn.hook_z'))
            for name,value in cache.items():
                contribution=value.sum(0,keepdim=True)
                if name not in sums[key]:
                    sums[key][name]=contribution.clone()
                else:
                    sums[key][name].add_(contribution)
            count+=len(batch)
            del logits,cache
        for value in sums[key].values():
            value.div_(count)
        counts[key]=count
        progress(f'ABC mean bank: group {i}/{len(by_group)}, {count} examples')
    metadata=dict(seed=seed,per_template_pattern=per_template,total_examples=len(examples),
        group_counts=[dict(key=list(k),n=v) for k,v in counts.items()],
        independent_of_evaluation=True,mean_by='template, pattern, exact length and semantic indices',
        examples=examples)
    return sums,metadata


def means_for_batch(bank,batch,device):
    keys=[group_key(ex) for ex in batch]
    missing=[k for k in keys if k not in bank]
    if missing:
        raise ValueError(f'No aligned reference mean for {missing[0]}')
    names=list(bank[keys[0]])
    return {name:torch.cat([bank[k][name] for k in keys],dim=0).to(device) for name in names}


@torch.inference_mode()
def evaluate_circuits(model, examples, bank, conditions, batch_size=16, progress=print,removals=None):
    device=next(model.parameters()).device
    removals={} if removals is None else removals
    if set(conditions).intersection(removals):
        raise ValueError('Condition names must be distinct')
    rows={name:[] for name in ['full','empty_attention',*conditions,*removals]}
    for i,batch in enumerate(batches(examples,batch_size),1):
        tokens=token_tensor(batch,'clean_ids',device)
        donor=means_for_batch(bank,batch,device)
        rows['full'].extend(score_rows(model(tokens),batch))
        runs=[(name,nodes,'keep') for name,nodes in {'empty_attention':[],**conditions}.items()]
        runs.extend((name,nodes,'remove') for name,nodes in removals.items())
        for name,nodes,mode in runs:
            logits=model.run_with_hooks(tokens,fwd_hooks=circuit_hooks(model,donor,batch,nodes,mode))
            rows[name].extend(score_rows(logits,batch))
            del logits
        progress(f'Circuit evaluation: batch {i}, {len(batch)} prompts')
    summaries={name:baseline_summary(values) for name,values in rows.items()}
    fidelity={}
    paired={}
    for name in conditions:
        fidelity[name]={}
        paired[name]={}
        for pattern in ('all','ABBA','BABA'):
            full,empty,retained=(summaries[k][pattern]['logit_diff']['mean'] for k in ('full','empty_attention',name))
            fidelity[name][pattern]=dict(fraction_of_full_logit_diff=retained/full if abs(full)>1e-6 else None,
                normalized_attention_recovery=(retained-empty)/(full-empty) if abs(full-empty)>1e-6 else None)
            full_rows=[r for r in rows['full'] if pattern=='all' or r['pattern']==pattern]
            kept_rows=[r for r in rows[name] if pattern=='all' or r['pattern']==pattern]
            paired[name][pattern]=stats(a['logit_diff']-b['logit_diff'] for a,b in zip(kept_rows,full_rows))
    return dict(summaries=summaries,fidelity=fidelity,paired_delta_vs_full=paired,rows=rows,
        scope='Attention-node circuits conditional on retained embeddings and MLPs; independent template-conditioned ABC means')


@torch.inference_mode()
def ov_copy_scores(model, names, layers=None, batch_names=16):
    """OV matrix copy probe adapted to raw BLOOM LayerNorm parameters.

    Source residual is the name position after block 0. Each head's own LN1
    is applied before V and O matrix composition. Projection biases are
    excluded from the OV matrix probe. Positive and negative head outputs
    are read through actual final LN and unembedding, separately.
    This is a copying capability test, not a claim about causal task use.
    """
    device=next(model.parameters()).device
    layers=range(model.cfg.n_layers) if layers is None else layers
    ids=[model.tokenizer.encode(' '+name,add_special_tokens=False)[0] for name in names]
    counts=defaultdict(lambda:[0,0,0])
    for start in range(0,len(ids),batch_names):
        target=torch.tensor(ids[start:start+batch_names],device=device)
        tokens=torch.stack([torch.full_like(target,model.tokenizer.bos_token_id),target],1)
        logits,cache=model.run_with_cache(tokens,names_filter=lambda n:n=='blocks.0.hook_resid_post')
        source=cache['blocks.0.hook_resid_post'][:,1]
        del logits,cache
        for layer in layers:
            block=model.blocks[layer]
            normalized=block.ln1(source)
            v=torch.einsum('bm,hmd->bhd',normalized,block.attn.W_V)
            result=torch.einsum('bhd,hdm->bhm',v,block.attn.W_O)
            for head in range(model.cfg.n_heads):
                for sign_index,sign in enumerate((1,-1)):
                    projected=model.ln_final(sign*result[:,head])@model.W_U+model.b_U
                    top=projected.topk(5,dim=-1).indices
                    counts[(layer,head)][sign_index]+=int((top==target[:,None]).any(-1).sum())
                    del projected
                counts[(layer,head)][2]+=len(target)
    return [dict(layer=l,head=h,n=v[2],positive_top5_copy_rate=v[0]/v[2],
                 negative_top5_copy_rate=v[1]/v[2]) for (l,h),v in sorted(counts.items())]
