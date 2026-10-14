#!/usr/bin/env python3
"""Real-model checks of Z/result equivalence, exact paths and LayerNorm readout.

Uses only cached BLOOM-560M weights; run separately from CPU unit tests.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformer_lens import TransformerBridge
from bloom_ioi.data import build_examples
from bloom_ioi.experiment import cache_filter
from bloom_ioi.interventions import Node, act_name, token_tensor, patch_head, path_patch, circuit_hooks, logit_diffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-json',type=Path,help='Save successful checks and numerical errors for experiment provenance')
    args = parser.parse_args()
    model = TransformerBridge.boot_transformers('bigscience/bloom-560m', device=args.device, dtype=torch.float32)
    model.eval()
    model.set_use_attn_result(True)
    examples = build_examples(model.tokenizer, 2, 42)[:2]
    tokens = token_tensor(examples, 'clean_ids', args.device)
    names = lambda n: cache_filter(n) or n.endswith(('attn.hook_result','ln1.hook_out')) or n == 'ln_final.hook_in'
    checks={}
    with torch.inference_mode():
        clean, cc = model.run_with_cache(tokens, names_filter=names)
        _, bc = model.run_with_cache(token_tensor(examples,'corrupt_ids',args.device),names_filter=names)
        projection_errors=[]
        for layer in (0,17,23):
            normalized=cc[f'blocks.{layer}.ln1.hook_out']
            for component in ('Q','K','V'):
                attention=model.blocks[layer].attn
                projected=torch.einsum('bpm,hmd->bphd',normalized,getattr(attention,'W_'+component))
                projected+=getattr(attention,'b_'+component)
                torch.testing.assert_close(projected,cc[act_name(layer,component.lower())],atol=5e-4,rtol=1e-4)
                projection_errors.append(float((projected-cc[act_name(layer,component.lower())]).abs().max()))
        checks['qkv_projection_layout']=dict(passed=True,max_abs_error=max(projection_errors),layers=[0,17,23])
        print('PASS: exposed Q/K/V weight matrices and biases reproduce actual cached head projections')
        sender = Node(17,9,'END')
        z_patched = model.run_with_hooks(tokens, fwd_hooks=[
            (act_name(sender.layer,'z'),patch_head(bc[act_name(sender.layer,'z')],examples,sender.head,sender.role))])
        result_patched = model.run_with_hooks(tokens, fwd_hooks=[
            (act_name(sender.layer,'result'),patch_head(bc[act_name(sender.layer,'result')],examples,sender.head,sender.role))])
        torch.testing.assert_close(z_patched,result_patched,atol=5e-4,rtol=1e-4)
        checks['z_result_equivalence']=dict(passed=True,max_abs_error=float((z_patched-result_patched).abs().max()))
        print('PASS: Z patch equals per-head residual-result patch')
        identity_errors=[]
        for receiver in (None, Node(20,10,'END'), [Node(18,7,'END'),Node(20,10,'END')]):
            identity,_ = path_patch(model,tokens,examples,cc,cc,sender,receiver,freeze_mlps=True)
            torch.testing.assert_close(identity,clean,atol=5e-4,rtol=1e-4)
            identity_errors.append(float((identity-clean).abs().max()))
        checks['identical_donor_paths']=dict(passed=True,max_abs_error=max(identity_errors))
        print('PASS: identical-donor output, individual and grouped receiver paths are identities')
        # With all MLPs and other heads frozen, only the sender's direct
        # residual contribution reaches final LayerNorm. Check in residual
        # space and read out through the original normalization and unembedding.
        direct,_ = path_patch(model,tokens,examples,cc,bc,sender,freeze_mlps=True)
        residual = cc['ln_final.hook_in'].clone()
        for i,ex in enumerate(examples):
            pos=ex['positions']['END']
            delta=bc[act_name(sender.layer,'z')][i,pos,sender.head]-cc[act_name(sender.layer,'z')][i,pos,sender.head]
            residual[i,pos] += delta @ model.blocks[sender.layer].attn.W_O[sender.head]
        expected = model.ln_final(residual) @ model.W_U + model.b_U
        torch.testing.assert_close(direct,expected,atol=5e-4,rtol=1e-4)
        checks['exact_frozen_mlp_direct_path']=dict(passed=True,max_abs_error=float((direct-expected).abs().max()))
        print('PASS: frozen-MLP head-to-logit path equals exact residual replacement with recomputed LayerNorm')
        # Ensure final-LN-aware projection, including affine bias, equals the
        # model's IO-S metric. This catches forgotten gamma/centering/scales.
        rows = torch.arange(len(examples),device=args.device)
        pos = torch.tensor([ex['positions']['END'] for ex in examples],device=args.device)
        r = cc['ln_final.hook_in'][rows,pos]
        ln = model.ln_final.original_component
        io = torch.tensor([ex['io_id'] for ex in examples],device=args.device)
        s = torch.tensor([ex['s_id'] for ex in examples],device=args.device)
        direction = (model.W_U[:,io]-model.W_U[:,s]).T
        projected = (((r-r.mean(-1,keepdim=True))/cc['ln_final.hook_scale'][rows,pos] * ln.weight + ln.bias)*direction).sum(-1)
        projected += model.b_U[io]-model.b_U[s]
        torch.testing.assert_close(projected,logit_diffs(clean,examples),atol=5e-4,rtol=1e-4)
        checks['affine_layernorm_readout']=dict(passed=True,max_abs_error=float((projected-logit_diffs(clean,examples)).abs().max()))
        print('PASS: centered/scaled/affine LayerNorm readout reproduces IO-S logits')
        full_nodes = [Node(l,h,'ALL') for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
        restored = model.run_with_hooks(tokens,fwd_hooks=circuit_hooks(model,bc,examples,full_nodes))
        torch.testing.assert_close(restored,clean,atol=5e-4,rtol=1e-4)
        checks['full_attention_retention']=dict(passed=True,max_abs_error=float((restored-clean).abs().max()))
        print('PASS: retaining every attention node exactly restores the full model')
    if args.output_json:
        args.output_json.parent.mkdir(parents=True,exist_ok=True)
        args.output_json.write_text(json.dumps(dict(status='complete',model='bigscience/bloom-560m',
            completed_utc=datetime.now(timezone.utc).isoformat(),device=args.device,dtype='float32',
            n_prompts=len(examples),atol=5e-4,rtol=1e-4,checks=checks,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')


if __name__ == '__main__':
    main()
