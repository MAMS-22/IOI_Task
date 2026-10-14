"""Exact activation interventions and frozen-head path patching.

Patches use z before W_O; replacing one z head is equivalent to replacing its
per-head residual contribution. Path isolation freezes other attention outputs
and optionally MLP outputs. Receiver Q/K/V is recorded BEFORE freezing outputs,
then installed in a separate otherwise clean forward pass (Easy-Transformer's
four-pass construction). No fixed-LayerNorm linear approximation is used.
"""
from dataclasses import dataclass
import math
import statistics
import torch


@dataclass(frozen=True, order=True)
class Node:
    layer: int
    head: int
    role: str

    def as_dict(self):
        return dict(layer=self.layer, head=self.head, role=self.role)


def act_name(layer, component):
    return f"blocks.{layer}.attn.hook_{component}"


def token_tensor(examples, key, device):
    return torch.tensor([ex[key] for ex in examples], dtype=torch.long, device=device)


def role_positions(examples, role, device):
    return torch.tensor([ex["positions"][role] for ex in examples], device=device)


def logit_diffs(logits, examples):
    logits = logits.logits if hasattr(logits, "logits") else logits
    rows = torch.arange(len(examples), device=logits.device)
    pos = role_positions(examples, "END", logits.device)
    io = torch.tensor([ex["io_id"] for ex in examples], device=logits.device)
    s = torch.tensor([ex["s_id"] for ex in examples], device=logits.device)
    scores = logits[rows, pos, io].float() - logits[rows, pos, s].float()
    if not torch.isfinite(scores).all():
        raise RuntimeError("Non-finite IOI scores")
    return scores


def patch_head(donor, examples, head, role):
    def patch(value, hook):
        value = value.clone()
        rows = torch.arange(len(examples), device=value.device)
        if role == "ALL":
            value[:, :, head, :] = donor[:, :, head, :]
        else:
            pos = role_positions(examples, role, value.device)
            value[rows, pos, head, :] = donor[rows, pos, head, :]
        return value
    return patch


def patch_residual(donor, examples, role):
    def patch(value, hook):
        value = value.clone()
        rows = torch.arange(len(examples), device=value.device)
        pos = role_positions(examples, role, value.device)
        value[rows, pos] = donor[rows, pos]
        return value
    return patch


def freeze_heads(model, clean_cache, corrupt_cache, examples, sender, freeze_mlps=False):
    hooks = []
    for layer in range(model.cfg.n_layers):
        name = act_name(layer, "z")
        def freeze(value, hook, layer=layer, name=name):
            fixed = clean_cache[name].clone()
            if layer == sender.layer:
                fixed = patch_head(corrupt_cache[name], examples, sender.head, sender.role)(fixed, hook)
            return fixed
        hooks.append((name, freeze))
        if freeze_mlps:
            name_mlp = f"blocks.{layer}.hook_mlp_out"
            def freeze_mlp(value, hook, name_mlp=name_mlp):
                return clean_cache[name_mlp].clone()
            hooks.append((name_mlp, freeze_mlp))
    return hooks


@torch.inference_mode()
def path_patch(model, tokens, examples, clean_cache, corrupt_cache, sender,
               receiver=None, component="q", freeze_mlps=False, observe=None):
    """Corrupt sender -> receiver input -> clean continuation, or directly logits.

    observe optionally names an attention-pattern hook to record in the final run.
    This measures how a candidate S-inhibition path changes mover attention.
    """
    if component not in ("q", "k", "v"):
        raise ValueError("receiver component must be q, k, or v")
    receivers = [] if receiver is None else ([receiver] if isinstance(receiver, Node) else list(receiver))
    if any(r.layer <= sender.layer for r in receivers):
        raise ValueError("Receiver must be in a later layer than sender.")
    hooks = freeze_heads(model, clean_cache, corrupt_cache, examples, sender, freeze_mlps)
    if not receivers:
        return model.run_with_hooks(tokens, fwd_hooks=hooks), {}
    captured = {}
    capture_hooks = []
    for name in sorted({act_name(r.layer, component) for r in receivers}):
        def capture(value, hook, name=name):
            captured[name] = value.detach().clone()
            return value
        capture_hooks.append((name, capture))
    # A receiver's qkv is computed from the changed residual BEFORE its z is
    # frozen; downstream heads never transmit its changed attention in this pass.
    model.run_with_hooks(tokens, fwd_hooks=capture_hooks + hooks)
    for name, _ in capture_hooks:
        if name not in captured:
            raise RuntimeError(f"Required receiver hook did not fire: {name}")
    final_cache = {}
    final_hooks = [(act_name(r.layer, component),
                    patch_head(captured[act_name(r.layer, component)], examples, r.head, r.role))
                   for r in receivers]
    if observe:
        def capture_pattern(value, hook):
            final_cache[observe] = value.detach().clone()
            return value
        final_hooks.append((observe, capture_pattern))
    return model.run_with_hooks(tokens, fwd_hooks=final_hooks), final_cache


def circuit_hooks(model, donor_cache, examples, nodes, mode="keep"):
    """Mean/resample ablate outside a node set, or ablate just that set."""
    if mode not in ("keep", "remove"):
        raise ValueError("mode must be keep or remove")
    hooks = []
    for layer in range(model.cfg.n_layers):
        name = act_name(layer, "z")
        selected = [node for node in nodes if node.layer == layer]
        def replace(value, hook, name=name, selected=selected):
            donor = donor_cache[name]
            result = donor.clone() if mode == "keep" else value.clone()
            source = value if mode == "keep" else donor
            for node in selected:
                result = patch_head(source, examples, node.head, node.role)(result, hook)
            return result
        hooks.append((name, replace))
    return hooks


def stats(values):
    values = [float(x) for x in values]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Expected nonempty finite measurements")
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return dict(n=len(values), mean=mean, standard_error=sem)


def effect_summary(clean, corrupt, patched, examples, direction="noising"):
    result = {}
    for pattern in ("all", "ABBA", "BABA"):
        idx = [i for i, ex in enumerate(examples) if pattern == "all" or ex['pattern'] == pattern]
        if not idx:
            continue
        c, b, p = ([float(v[i]) for i in idx] for v in (clean, corrupt, patched))
        gap = statistics.mean(c) - statistics.mean(b)
        delta = [x - y for x, y in zip(p, c if direction == "noising" else b)]
        result[pattern] = dict(clean=stats(c), corrupt=stats(b), patched=stats(p),
                               delta=stats(delta), clean_corrupt_gap=gap,
                               normalized_effect=statistics.mean(delta) / gap if abs(gap) > 1e-6 else None)
    return result
