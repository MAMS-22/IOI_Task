#!/usr/bin/env python3
"""Reproducible English BLOOM IOI circuit discovery and validation.

Run --help or read English_Task/README.md for stages and interpretation limits.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

from bloom_ioi.data import build_examples, TEMPLATE_SOURCE, ROLES
from bloom_ioi.interventions import Node, act_name, token_tensor
from bloom_ioi.experiment import Experiment, select_nodes

STAGES = ('baseline', 'residual', 'heads', 'direct', 'paths', 'functions', 'validate')


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def export_dot(path, result):
    lines = ['digraph IOI {', 'rankdir=LR;', 'logits [shape=box,label="IO minus S logit"];']
    nodes = [Node(**n) for n in result.get('candidate_nodes', [])]
    def node_id(n):
        return f'L{n.layer}H{n.head}_{n.role.replace("+", "plus")}'
    for n in nodes:
        lines.append(f'{node_id(n)} [label="{n.layer}.{n.head} @ {n.role}"];')
    for record in result.get('direct', []):
        n = Node(**record['node'])
        if n in nodes:
            effect = record['effects']['all']['delta']['mean']
            lines.append(f'{node_id(n)} -> logits [label="delta {effect:+.3f}"];')
    for record in result.get('confirmed_edges', []):
        sender, receiver = Node(**record['sender']), Node(**record['receiver'])
        # Receiver input location can differ from its retained output location.
        for source in (sender, receiver):
            lines.append(f'{node_id(source)} [label="{source.layer}.{source.head} @ {source.role}"];')
        effect = record['effects']['all']['delta']['mean']
        lines.append(f'{node_id(sender)} -> {node_id(receiver)} [label="{record["component"]}: {effect:+.3f}"];')
    lines.append('}')
    path.write_text('\n'.join(lines) + '\n')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='bigscience/bloom-560m')
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-per-pattern', type=int, default=200, help='Baseline examples per pattern')
    p.add_argument('--discovery-n', type=int, default=50, help='Discovery examples per pattern')
    p.add_argument('--validation-n', type=int, default=200, help='Independent held-out examples per pattern')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--corruption', choices=('abc', 'subject_swap'), default='abc')
    p.add_argument('--no-bos', action='store_true')
    p.add_argument('--freeze-mlps', action='store_true', help='Exclude MLP-mediated paths during path tracing')
    p.add_argument('--stages', default=','.join(STAGES), help='Comma-separated stage names')
    p.add_argument('--layers', help='Comma-separated zero-based layers; default scans every layer')
    p.add_argument('--head-roles', default=','.join(ROLES), help='Positions for denoising head scan')
    p.add_argument('--top-k', type=int, default=8, help='Candidate count per backward-tracing tier; output tier keeps each sign')
    p.add_argument('--min-effect', type=float, default=0.02, help='Minimum absolute raw logit effect for candidates')
    p.add_argument('--routing-threshold', type=float, default=0.1, help='IOI attention screen for proposed roles; 0 disables guidance')
    p.add_argument('--mean-donors', type=int, default=8)
    p.add_argument('--random-trials', type=int, default=3)
    p.add_argument('--probe-samples', type=int, default=32)
    p.add_argument('--probe-length', type=int, default=32)
    p.add_argument('--offline', action='store_true', help='Use only cached model/tokenizer files')
    p.add_argument('--verify-hf', action='store_true', help='Compare bridge to a separate unmodified Hugging Face model')
    p.add_argument('--circuit', type=Path, help='JSON array of {layer,head,role} nodes, or previous result JSON')
    p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'english_results' / 'circuit_discovery')
    p.add_argument('--resume', action='store_true', help='Skip completed stages in an identical-config run')
    return p


def main():
    p = parser()
    args = p.parse_args()
    stages = args.stages.split(',')
    if any(s not in STAGES for s in stages) or len(set(stages)) != len(stages):
        p.error(f'stages must be distinct members of {STAGES}')
    # Always execute dependencies in their natural order.
    stages = [s for s in STAGES if s in stages]
    for name in ('n_per_pattern', 'discovery_n', 'validation_n', 'batch_size', 'top_k',
                 'mean_donors', 'random_trials', 'probe_samples'):
        if getattr(args, name) < 1:
            p.error(f'{name} must be positive')
    if args.probe_length < 2 or args.min_effect < 0 or not 0 <= args.routing_threshold <= 1:
        p.error('probe-length must be >=2; min-effect >=0; routing-threshold in [0,1]')
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from transformer_lens import TransformerBridge
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    # Preserve original BLOOM logits and learned LayerNorm scales for attribution.
    model = TransformerBridge.boot_transformers(args.model, device=device, dtype=torch.float32)
    model.eval()
    if model.cfg.architecture != 'BloomForCausalLM':
        p.error('This implementation validates BLOOM hooks and LayerNorm; use a BLOOM checkpoint.')
    layers = list(range(model.cfg.n_layers)) if args.layers is None else [int(x) for x in args.layers.split(',')]
    if not layers or len(set(layers)) != len(layers) or any(l < 0 or l >= model.cfg.n_layers for l in layers):
        p.error('layers must be unique valid zero-based indices')
    roles = args.head_roles.split(',')
    if not roles or len(set(roles)) != len(roles) or any(r not in ROLES and r != 'ALL' for r in roles):
        p.error(f'head-roles must be members of {ROLES} or ALL')
    baseline = build_examples(model.tokenizer, args.n_per_pattern, args.seed, args.corruption, not args.no_bos)
    discovery = build_examples(model.tokenizer, args.discovery_n, args.seed, args.corruption, not args.no_bos)
    excluded = {ex['prompt'] for ex in baseline + discovery}
    heldout = build_examples(model.tokenizer, args.validation_n, args.seed + 1, args.corruption,
                             not args.no_bos, exclude_prompts=excluded)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ('resume', 'stages', 'output_dir', 'verify_hf')}
    config.update(device=device, layers=layers)
    output = args.output_dir / 'results.json'
    if args.resume and output.exists():
        result = json.loads(output.read_text())
        if result['config'] != config:
            p.error('Resume configuration differs from saved run; choose another output directory')
    else:
        if output.exists():
            p.error('Output exists; use --resume or a new --output-dir to preserve earlier results')
        result = dict(schema_version=1, config=config, started_utc=datetime.now(timezone.utc).isoformat(),
            versions={n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'transformer_lens')},
            template_source=TEMPLATE_SOURCE,
            reproduction_scope='English path-patching branch of Zhang et al.; lexical pools and sampling are not author-verified',
            patching_method='Exact frozen attention-output four-pass path patching; MLPs ' + ('frozen' if args.freeze_mlps else 'unfrozen (MLP-mediated paths included)'),
            metric='next-token logit(IO) minus logit(S), always scored on clean targets; answer excluded from input',
            discovery_selection='raw signed mean logit changes, separate output-head signs, optional IOI routing screens; no held-out selection',
            baseline_examples=baseline, discovery_examples=discovery, validation_examples=heldout,
            completed_stages=[], candidate_nodes=[], confirmed_edges=[])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Fail early if aliases exist but are nonfunctional or have unexpected layouts.
    first = discovery[:1]
    tokens = token_tensor(first, 'clean_ids', device)
    required = {act_name(l, c) for l in layers for c in ('z', 'q', 'k', 'v', 'pattern')}
    with torch.inference_mode():
        logits, cache = model.run_with_cache(tokens, names_filter=lambda n: n in required or n == 'ln_final.hook_scale')
    missing = required.difference(cache.keys())
    if missing:
        raise RuntimeError(f'Required hooks did not fire: {sorted(missing)}')
    for l in layers:
        expected = (1, tokens.shape[1], model.cfg.n_heads, model.cfg.d_head)
        for c in ('z', 'q', 'k', 'v'):
            if tuple(cache[act_name(l, c)].shape) != expected:
                raise RuntimeError(f'Unexpected {l}/{c} layout')
    result.setdefault('hook_verification', {}).update(
        required_hooks_fired=len(required), logits_finite=bool(torch.isfinite(logits).all()))
    if not result['hook_verification']['logits_finite']:
        raise RuntimeError('Non-finite logits')
    if args.verify_hf:
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                  local_files_only=args.offline).to(device).eval()
        with torch.inference_mode():
            expected_logits = hf(tokens).logits
        maximum = float((logits - expected_logits).abs().max())
        torch.testing.assert_close(logits, expected_logits, atol=5e-4, rtol=1e-4)
        result['hook_verification']['hf_max_abs_logit_error'] = maximum
        del hf, expected_logits
    del logits, cache
    exp = Experiment(model, args.batch_size, layers, args.freeze_mlps, lambda msg: print(msg, flush=True))
    if args.circuit:
        supplied = json.loads(args.circuit.read_text())
        result['candidate_nodes'] = supplied if isinstance(supplied, list) else supplied['candidate_nodes']
        for n in result['candidate_nodes']:
            node = Node(**n)
            if node.role not in ROLES and node.role != 'ALL':
                p.error('Invalid circuit role')
            if not (0 <= node.layer < model.cfg.n_layers and 0 <= node.head < model.cfg.n_heads):
                p.error('Invalid circuit layer/head')
    atomic_json(output, result)

    def pick(records, sign='absolute', k=None, tier=None):
        if tier and args.routing_threshold > 0:
            evidence = {(x['layer'], x['head']): x for x in result['selection_evidence']}
            screened = []
            for record in records:
                node = Node(**record['node'])
                measurements = evidence[(node.layer, node.head)]['ioi']
                if tier == 'output':
                    passed = (measurements['attention_END_to_IO']['mean'] >= args.routing_threshold
                              and abs(measurements['end_logit_diff_attribution']['mean']) >= args.min_effect)
                elif tier == 'query':
                    passed = measurements['attention_END_to_S2']['mean'] >= args.routing_threshold
                elif tier == 'value':
                    passed = max(measurements['attention_S2_to_S1']['mean'],
                                 measurements['attention_S2_to_S1+1']['mean']) >= args.routing_threshold
                elif tier == 'key':
                    passed = measurements['attention_S1+1_to_S1']['mean'] >= args.routing_threshold
                else:
                    raise ValueError(tier)
                if passed:
                    screened.append(record)
            records = screened
        return select_nodes(records, k or args.top_k, sign, args.min_effect)

    for stage in stages:
        if stage in result['completed_stages']:
            print(f'Skipping completed {stage}', flush=True)
            continue
        print(f'Starting {stage}', flush=True)
        started = time.monotonic()
        if stage == 'baseline':
            result[stage] = exp.baseline(baseline)
            print(json.dumps(result[stage]['clean'], indent=2), flush=True)
        elif stage == 'residual':
            result[stage] = exp.scan(discovery, 'residual', ROLES)
        elif stage == 'heads':
            result[stage] = exp.scan(discovery, 'heads', roles)
        elif stage == 'direct':
            result[stage] = exp.scan(discovery, 'direct')
            if args.routing_threshold > 0:
                result['selection_evidence'] = exp.functional_evidence(discovery, args.seed + 100,
                    args.probe_samples, args.probe_length)
            # Noising an important positive contributor gives a negative change;
            # a negative contributor gives a positive change. Keep both.
            movers = pick(result[stage], 'negative', tier='output') + pick(result[stage], 'positive', tier='output')
            result['output_candidates'] = [n.as_dict() for n in movers]
            result['candidate_nodes'] = [n.as_dict() for n in sorted(set(movers))]
        elif stage == 'paths':
            if not result.get('output_candidates'):
                p.error('paths requires direct results with output candidates; run direct first')
            movers = [Node(**n) for n in result['output_candidates']]
            # A restarted partial paths stage recomputes all tiers. Avoid
            # duplicating individual edges from its previous checkpoint.
            result['confirmed_edges'] = []
            tiers = [dict(name='query_paths', role='END', receivers=movers, component='q'),
                     dict(name='value_paths', role='S2', receivers=None, component='v'),
                     dict(name='key_paths', role='S1+1', receivers=None, component='k')]
            current = movers
            all_nodes = set(Node(**n) for n in result['candidate_nodes'])
            for tier in tiers:
                if tier['name'] == 'value_paths':
                    receivers = [Node(n.layer, n.head, 'S2') for n in current]
                elif tier['name'] == 'key_paths':
                    if args.routing_threshold > 0:
                        evidence = {(x['layer'],x['head']): x for x in result['selection_evidence']}
                        # Previous-token information enters induction K at S1+1;
                        # duplicate heads instead read the S1 name token itself.
                        induction = [n for n in current if
                            evidence[(n.layer,n.head)]['ioi']['attention_S2_to_S1+1']['mean'] >= args.routing_threshold]
                    else:
                        induction = current
                    receivers = [Node(n.layer, n.head, 'S1+1') for n in induction]
                else:
                    receivers = current
                records = exp.scan(discovery, 'path', (tier['role'],), receivers, tier['component'])
                selected = pick(records, tier={'query_paths':'query','value_paths':'value','key_paths':'key'}[tier['name']])
                result[tier['name']] = dict(records=records, selected=[n.as_dict() for n in selected],
                    note='Grouped receiver screen; individual sender/receiver edges confirmed separately')
                if selected:
                    result['confirmed_edges'].extend(exp.confirm_edges(discovery, selected, receivers, tier['component']))
                    all_nodes.update(selected)
                current = selected
                result['candidate_nodes'] = [n.as_dict() for n in sorted(all_nodes)]
                atomic_json(output, result)
                if not current:
                    break
        elif stage == 'functions':
            result[stage] = result.get('selection_evidence') or exp.functional_evidence(
                discovery, args.seed + 100, args.probe_samples, args.probe_length)
        elif stage == 'validate':
            if not result['candidate_nodes']:
                p.error('validate requires discovered or --circuit supplied nodes')
            result[stage] = exp.validate(heldout, [Node(**n) for n in result['candidate_nodes']],
                args.seed + 2, args.mean_donors, args.random_trials)
            print(json.dumps(result[stage]['fidelity'], indent=2), flush=True)
        result['completed_stages'].append(stage)
        result.setdefault('stage_seconds', {})[stage] = time.monotonic() - started
        result['updated_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(output, result)
        export_dot(args.output_dir / 'candidate_circuit.dot', result)
    print(f'Saved {output.resolve()}', flush=True)
    print('Candidate functions require converging causal and probe evidence. Inspect held-out fidelity before treating the node set as a circuit.', flush=True)


if __name__ == '__main__':
    main()
