"""Discovery, functional evidence, and independent circuit validation."""
from collections import defaultdict
import random
import torch

from .data import batches, name_pool, abc_control
from .interventions import (Node, act_name, token_tensor, role_positions, logit_diffs,
                            patch_head, patch_residual, path_patch, circuit_hooks,
                            stats, effect_summary)


def cache_filter(name):
    return (name.endswith(('attn.hook_z', 'attn.hook_q', 'attn.hook_k', 'attn.hook_v',
                           'attn.hook_pattern', 'hook_resid_pre', 'hook_mlp_out'))
            or name == 'ln_final.hook_scale')


def score_rows(logits, examples):
    diffs = logit_diffs(logits, examples).tolist()
    rows = torch.arange(len(examples), device=logits.device)
    next_logits = logits[rows, role_positions(examples, 'END', logits.device)]
    top = next_logits.argmax(-1).tolist()
    return [dict(example_id=ex['example_id'], pattern=ex['pattern'], template_id=ex['template_id'],
                 logit_diff=diff, correct=diff > 0, zero_rank=prediction == ex['io_id'],
                 top_token_id=prediction)
            for ex, diff, prediction in zip(examples, diffs, top)]


def baseline_summary(rows):
    result = {}
    for pattern in ('all', 'ABBA', 'BABA'):
        group = [r for r in rows if pattern == 'all' or r['pattern'] == pattern]
        if group:
            result[pattern] = dict(logit_diff=stats(r['logit_diff'] for r in group),
                                   accuracy=sum(r['correct'] for r in group) / len(group),
                                   zero_rank_rate=sum(r['zero_rank'] for r in group) / len(group))
    return result


def attention_at(pattern, examples, head, dest, source):
    rows = torch.arange(len(examples), device=pattern.device)
    q, k = (role_positions(examples, role, pattern.device) for role in (dest, source))
    return pattern[rows, head, q, k]


class Experiment:
    def __init__(self, model, batch_size=8, layers=None, freeze_mlps=False, progress=print):
        self.model = model
        self.device = next(model.parameters()).device
        self.batch_size = batch_size
        self.layers = list(range(model.cfg.n_layers)) if layers is None else layers
        self.freeze_mlps = freeze_mlps
        self.progress = progress

    @torch.inference_mode()
    def baseline(self, examples):
        clean, corrupt = [], []
        for batch in batches(examples, self.batch_size):
            clean.extend(score_rows(self.model(token_tensor(batch, 'clean_ids', self.device)), batch))
            corrupt.extend(score_rows(self.model(token_tensor(batch, 'corrupt_ids', self.device)), batch))
        return dict(clean=baseline_summary(clean), corrupt_on_clean_targets=baseline_summary(corrupt),
                    clean_rows=clean, corrupt_rows=corrupt)

    @torch.inference_mode()
    def scan(self, examples, kind, roles=('END',), receivers=(), component='q', sender_nodes=None,
             retain_values=False):
        """Collect scalar per-example effects, keeping just one activation batch alive."""
        candidates = [Node(l, h, role) for l in self.layers for h in range(self.model.cfg.n_heads)
                      for role in roles]
        if kind == 'residual':
            candidates = [Node(l, -1, role) for l in self.layers for role in roles]
        if sender_nodes is not None:
            candidates = list(sender_nodes)
        if kind == 'path':
            if not receivers:
                return []
            # A later sender can still feed later receivers even if some
            # members of the receiver group occur before it.
            candidates = [n for n in candidates if n.layer < max(r.layer for r in receivers)]
        values = {node: [] for node in candidates}
        clean, corrupt, ordered_examples = [], [], []
        for bi, batch in enumerate(batches(examples, self.batch_size), 1):
            self.progress(f'{kind}: batch {bi}, {len(batch)} examples, {len(candidates)} interventions')
            tokens = token_tensor(batch, 'clean_ids', self.device)
            corrupt_tokens = token_tensor(batch, 'corrupt_ids', self.device)
            clean_logits, clean_cache = self.model.run_with_cache(tokens, names_filter=cache_filter)
            corrupt_logits, corrupt_cache = self.model.run_with_cache(corrupt_tokens, names_filter=cache_filter)
            clean.extend(logit_diffs(clean_logits, batch).tolist())
            corrupt.extend(logit_diffs(corrupt_logits, batch).tolist())
            del clean_logits, corrupt_logits
            ordered_examples.extend(batch)
            for node in candidates:
                if kind in ('direct', 'path'):
                    logits, _ = path_patch(self.model, tokens, batch, clean_cache, corrupt_cache,
                                           node, [r for r in receivers if r.layer > node.layer] if kind == 'path' else None,
                                           component, self.freeze_mlps)
                elif kind == 'residual':
                    name = f'blocks.{node.layer}.hook_resid_pre'
                    logits = self.model.run_with_hooks(corrupt_tokens,
                        fwd_hooks=[(name, patch_residual(clean_cache[name], batch, node.role))])
                elif kind == 'heads':
                    name = act_name(node.layer, 'z')
                    logits = self.model.run_with_hooks(corrupt_tokens,
                        fwd_hooks=[(name, patch_head(clean_cache[name], batch, node.head, node.role))])
                else:
                    raise ValueError(f'Unknown scan {kind}')
                values[node].extend(logit_diffs(logits, batch).tolist())
                del logits
            del clean_cache, corrupt_cache
        records = [dict(node=node.as_dict(), receivers=[r.as_dict() for r in receivers if r.layer > node.layer], component=component,
                     direction='noising' if kind in ('direct', 'path') else 'denoising',
                     effects=effect_summary(clean, corrupt, values[node], ordered_examples,
                         'noising' if kind in ('direct', 'path') else 'denoising'))
                for node in candidates]
        if retain_values:
            for record,node in zip(records,candidates):
                reference = clean if kind in ('direct','path') else corrupt
                record['per_example_delta'] = [p-b for p,b in zip(values[node],reference)]
                record['example_ids'] = [ex['example_id'] for ex in ordered_examples]
        return records

    @torch.inference_mode()
    def functional_evidence(self, examples, random_seed=123, random_samples=32, sequence_length=32,
                            include_scatter=False):
        """Routing + final-LN-aware DLA + independent repeated-random-token probes.

        Attention and DLA are evidence about behavior; they do not by themselves
        assign a causal function or implement Ferrando & Voita's flow routes.
        """
        data = defaultdict(lambda: defaultdict(list))
        for batch in batches(examples, self.batch_size):
            tokens = token_tensor(batch, 'clean_ids', self.device)
            _, cache = self.model.run_with_cache(tokens, names_filter=cache_filter)
            rows = torch.arange(len(batch), device=self.device)
            end = role_positions(batch, 'END', self.device)
            a = torch.tensor([ex['io_id'] for ex in batch], device=self.device)
            b = torch.tensor([ex['s_id'] for ex in batch], device=self.device)
            # Raw bridge weights retain learned final LayerNorm gamma.
            ln = self.model.ln_final
            gamma = getattr(ln, 'w', None)
            if gamma is None:
                gamma = ln.original_component.weight
            io_direction = self.model.W_U[:, a].T.float() * gamma.float()
            s_direction = self.model.W_U[:, b].T.float() * gamma.float()
            direction = io_direction-s_direction
            scale = cache['ln_final.hook_scale'][rows, end].float()
            for layer in self.layers:
                pattern = cache[act_name(layer, 'pattern')]
                z_end = cache[act_name(layer, 'z')][rows, end].float()
                # Local W_O avoids repeatedly stacking all 24 layers' weights.
                result = torch.einsum('bhd,hdm->bhm', z_end, self.model.blocks[layer].attn.W_O.float())
                result = result - result.mean(-1, keepdim=True)
                dla = ((result / scale[:, None, :]) * direction[:, None, :]).sum(-1)
                io_dla = ((result / scale[:, None, :]) * io_direction[:, None, :]).sum(-1)
                s_dla = ((result / scale[:, None, :]) * s_direction[:, None, :]).sum(-1)
                for head in range(self.model.cfg.n_heads):
                    measurements = {'end_logit_diff_attribution': dla[:, head],
                                    'end_IO_logit_attribution':io_dla[:,head],
                                    'end_S_logit_attribution':s_dla[:,head]}
                    for dest, source in (('END', 'IO'), ('END', 'S1'), ('END', 'S2'),
                                         ('S2', 'S1'), ('S2', 'S1+1'), ('S1+1', 'S1')):
                        measurements[f'attention_{dest}_to_{source}'] = attention_at(pattern, batch, head, dest, source)
                    for label, values in measurements.items():
                        for ex, value in zip(batch, values.tolist()):
                            data[(layer, head)][label].append(value)
                            data[(layer, head)][label + '_' + ex['pattern']].append(value)
            del cache
        probe = self.random_token_probes(random_seed, random_samples, sequence_length)
        records=[dict(layer=l, head=h, ioi={label: stats(v) for label, v in data[(l, h)].items()},
                     random_tokens=probe[(l, h)]) for l in self.layers for h in range(self.model.cfg.n_heads)]
        if include_scatter:
            import statistics
            for record in records:
                values=data[(record['layer'],record['head'])]
                io=values['attention_END_to_IO']
                s=[a+b for a,b in zip(values['attention_END_to_S1'],values['attention_END_to_S2'])]
                io_projection=values['end_IO_logit_attribution']
                s_projection=values['end_S_logit_attribution']
                def corr(x,y):
                    return statistics.correlation(x,y) if statistics.pstdev(x)>1e-12 and statistics.pstdev(y)>1e-12 else None
                record['copying_scatter']=dict(io_attention=io,s_attention=s,io_projection=io_projection,
                    s_projection=s_projection,io_pearson=corr(io,io_projection),s_pearson=corr(s,s_projection))
        return records

    @torch.inference_mode()
    def random_token_probes(self, seed, n, length):
        if n < 1 or length < 2 or 2 * length + 1 > self.model.cfg.n_ctx:
            raise ValueError('Invalid random token probe size')
        rng = random.Random(seed)
        special = set(self.model.tokenizer.all_special_ids)
        pool = [i for i in range(len(self.model.tokenizer)) if i not in special]
        if length > len(pool):
            raise ValueError('Random sequence length exceeds token vocabulary')
        values = defaultdict(lambda: defaultdict(list))
        bos = self.model.tokenizer.bos_token_id
        # An explicit BOS keeps indices identical to the paper's random-token probes.
        if bos is None:
            raise ValueError('Random-token probe requires a BOS token')
        for start in range(0, n, self.batch_size):
            seqs = [rng.sample(pool, length) for _ in range(min(self.batch_size, n - start))]
            repeated = torch.tensor([[bos] + s + s for s in seqs], device=self.device)
            unique = torch.tensor([[bos] + rng.sample(pool, 2 * length) for _ in seqs], device=self.device)
            for condition, tokens in (('repeated', repeated), ('unique', unique)):
                _, cache = self.model.run_with_cache(tokens, names_filter=lambda name: name.endswith('attn.hook_pattern'))
                for layer in self.layers:
                    p = cache[act_name(layer, 'pattern')]
                    q = torch.arange(length + 1, 2 * length + 1, device=self.device)
                    measurements = dict(
                        duplicate=p[:, :, q, q - length].mean(-1),
                        induction=p[:, :, q[:-1], q[:-1] - length + 1].mean(-1),
                        previous=p[:, :, q, q - 1].mean(-1))
                    for head in range(self.model.cfg.n_heads):
                        for label, v in measurements.items():
                            values[(layer, head)][label + '_' + condition].extend(v[:, head].tolist())
                del cache
        return {key: {label: stats(v) for label, v in per_head.items()} for key, per_head in values.items()}

    @torch.inference_mode()
    def confirm_edges(self, examples, senders, receivers, component):
        """Individual edges and receiver attention changes after path intervention."""
        measurements = []
        for sender in senders:
            for receiver in receivers:
                if sender.layer >= receiver.layer:
                    continue
                c, b, p, attention_delta, ordered = [], [], [], [], []
                for batch in batches(examples, self.batch_size):
                    tokens = token_tensor(batch, 'clean_ids', self.device)
                    clean_logits, cc = self.model.run_with_cache(tokens, names_filter=cache_filter)
                    corrupt_logits, bc = self.model.run_with_cache(token_tensor(batch, 'corrupt_ids', self.device), names_filter=cache_filter)
                    c.extend(logit_diffs(clean_logits, batch).tolist())
                    b.extend(logit_diffs(corrupt_logits, batch).tolist())
                    name = act_name(receiver.layer, 'pattern')
                    logits, pc = path_patch(self.model, tokens, batch, cc, bc, sender, receiver,
                                           component, self.freeze_mlps, observe=name)
                    p.extend(logit_diffs(logits, batch).tolist())
                    old = attention_at(cc[name], batch, receiver.head, 'END', 'IO') - attention_at(cc[name], batch, receiver.head, 'END', 'S1')
                    new = attention_at(pc[name], batch, receiver.head, 'END', 'IO') - attention_at(pc[name], batch, receiver.head, 'END', 'S1')
                    attention_delta.extend((new - old).tolist())
                    ordered.extend(batch)
                    del cc, bc, pc, logits, clean_logits, corrupt_logits
                measurements.append(dict(sender=sender.as_dict(), receiver=receiver.as_dict(), component=component,
                    effects=effect_summary(c, b, p, ordered),
                    delta_attention_END_IO_minus_S1=stats(attention_delta)))
        return measurements

    @torch.inference_mode()
    def confirm_edges_batched(self,examples,senders,receivers,component):
        """Same individual edge tests, caching each prompt batch only once."""
        edges=[(s,r) for s in senders for r in receivers if s.layer<r.layer]
        if not edges:
            return []
        scores={edge:[] for edge in edges}
        attentions={edge:[] for edge in edges}
        clean,corrupt,ordered=[],[],[]
        for i,batch in enumerate(batches(examples,self.batch_size),1):
            self.progress(f'Confirm individual edges: batch {i}, {len(edges)} edges')
            tokens=token_tensor(batch,'clean_ids',self.device)
            clean_logits,cc=self.model.run_with_cache(tokens,names_filter=cache_filter)
            corrupt_logits,bc=self.model.run_with_cache(token_tensor(batch,'corrupt_ids',self.device),names_filter=cache_filter)
            clean.extend(logit_diffs(clean_logits,batch).tolist())
            corrupt.extend(logit_diffs(corrupt_logits,batch).tolist())
            ordered.extend(batch)
            del clean_logits,corrupt_logits
            for edge in edges:
                sender,receiver=edge
                name=act_name(receiver.layer,'pattern')
                logits,pc=path_patch(self.model,tokens,batch,cc,bc,sender,receiver,component,
                                     self.freeze_mlps,observe=name)
                scores[edge].extend(logit_diffs(logits,batch).tolist())
                old=attention_at(cc[name],batch,receiver.head,'END','IO')-attention_at(cc[name],batch,receiver.head,'END','S1')
                new=attention_at(pc[name],batch,receiver.head,'END','IO')-attention_at(pc[name],batch,receiver.head,'END','S1')
                attentions[edge].extend((new-old).tolist())
                del logits,pc
            del cc,bc
        return [dict(node=sender.as_dict(),sender=sender.as_dict(),receiver=receiver.as_dict(),component=component,
            effects=effect_summary(clean,corrupt,scores[(sender,receiver)],ordered),
            delta_attention_END_IO_minus_S1=stats(attentions[(sender,receiver)])) for sender,receiver in edges]

    @torch.inference_mode()
    def validate(self, examples, nodes, seed=44, donors=8, random_trials=3):
        """Independent mean-ABC controls and attention-node sufficiency/minimality.

        Embeddings and MLPs remain intact; this is not validation of a full
        attention+MLP circuit. Random controls match layer and role counts.
        """
        if donors < 1 or random_trials < 1:
            raise ValueError('Positive donor and random trial counts required')
        rng = random.Random(seed)
        names = name_pool(self.model.tokenizer)
        nodes = sorted(set(nodes))
        random_sets = []
        for _ in range(random_trials):
            by_layer_role = defaultdict(list)
            for node in nodes:
                by_layer_role[(node.layer, node.role)].append(node)
            random_nodes = []
            for (l, role), group in by_layer_role.items():
                random_nodes.extend(Node(l, h, role) for h in rng.sample(range(self.model.cfg.n_heads), len(group)))
            random_sets.append(random_nodes)
        conditions = {'full': [], 'empty_attention': [], 'candidate_only': [], 'candidate_removed': []}
        conditions.update({f'leave_out_{i}': [] for i in range(len(nodes))})
        conditions.update({f'random_{i}': [] for i in range(random_trials)})
        for bi, batch in enumerate(batches(examples, self.batch_size), 1):
            self.progress(f'validate: batch {bi}, {len(batch)} examples')
            tokens = token_tensor(batch, 'clean_ids', self.device)
            mean_cache = {}
            # Every donor preserves the held-out example's exact non-name context.
            # No discovery activations or score-based donor selection are used.
            for _ in range(donors):
                ids = [abc_control(ex, self.model.tokenizer, rng, names) for ex in batch]
                _, cache = self.model.run_with_cache(torch.tensor(ids, device=self.device),
                    names_filter=lambda name: name.endswith('attn.hook_z'))
                for name, value in cache.items():
                    if name not in mean_cache:
                        mean_cache[name] = value.clone() / donors
                    else:
                        mean_cache[name].add_(value / donors)
                del cache
            conditions['full'].extend(score_rows(self.model(tokens), batch))
            runs = [('empty_attention', [], 'keep'), ('candidate_only', nodes, 'keep'),
                    ('candidate_removed', nodes, 'remove')]
            runs.extend((f'leave_out_{i}', [n for n in nodes if n != node], 'keep') for i, node in enumerate(nodes))
            runs.extend((f'random_{i}', group, 'keep') for i, group in enumerate(random_sets))
            for label, group, mode in runs:
                logits = self.model.run_with_hooks(tokens,
                    fwd_hooks=circuit_hooks(self.model, mean_cache, batch, group, mode))
                conditions[label].extend(score_rows(logits, batch))
                del logits
            del mean_cache
        summaries = {label: baseline_summary(rows) for label, rows in conditions.items()}
        fidelity = {}
        for pattern in summaries['full']:
            full, empty, candidate = (summaries[label][pattern]['logit_diff']['mean']
                                       for label in ('full', 'empty_attention', 'candidate_only'))
            fidelity[pattern] = dict(full_minus_empty=full - empty,
                fraction_of_full_logit_diff=candidate / full if abs(full) > 1e-6 else None,
                normalized_attention_recovery=(candidate - empty) / (full - empty) if abs(full - empty) > 1e-6 else None)
        removals = []
        for i, node in enumerate(nodes):
            effects = {}
            for pattern in summaries['candidate_only']:
                base_rows = [r for r in conditions['candidate_only'] if pattern == 'all' or r['pattern'] == pattern]
                ablated_rows = [r for r in conditions[f'leave_out_{i}'] if pattern == 'all' or r['pattern'] == pattern]
                effects[pattern] = stats(a['logit_diff'] - b['logit_diff'] for a, b in zip(base_rows, ablated_rows))
            removals.append(dict(node=node.as_dict(), drop_from_candidate=effects))
        return dict(scope='attention nodes only; all embeddings and MLPs retained', replacement='mean of disjoint ABC donors per held-out context',
                    donor_count=donors, seed=seed, summaries=summaries, fidelity=fidelity,
                    leave_one_out=removals, random_nodes=[[n.as_dict() for n in group] for group in random_sets],
                    rows=conditions)


def select_nodes(records, top_k, sign='absolute', min_effect=0.02):
    """Discovery uses raw signed logit changes, not unstable normalized rankings."""
    ranked = []
    for record in records:
        effect = record['effects']['all']['delta']['mean']
        if abs(effect) < min_effect or (sign == 'positive' and effect <= 0) or (sign == 'negative' and effect >= 0):
            continue
        ranked.append((abs(effect), Node(**record['node'])))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]))
    return [node for _, node in ranked[:top_k]]
