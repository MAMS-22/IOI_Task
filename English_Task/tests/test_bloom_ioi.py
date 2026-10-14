"""CPU regression tests for alignment, effect signs and causal isolation.

Run: python -m unittest discover -s English_Task/tests -v
"""
import ast
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bloom_ioi.data import BABA_TEMPLATES, build_examples, batches
from bloom_ioi.interventions import (Node, act_name, patch_head, path_patch,
                                     circuit_hooks, effect_summary, logit_diffs)
from bloom_ioi.experiment import Experiment, select_nodes
from bloom_ioi.reproduction import select_supported,group_key,independent_means,means_for_batch


class WordTokenizer:
    bos_token_id = 1
    def __init__(self):
        self.vocab = {}
    def __call__(self, text, **kwargs):
        matches = list(re.finditer(r'\w+|[^\w\s]', text))
        ids = [self.vocab.setdefault(m.group(), len(self.vocab) + 4) for m in matches]
        return dict(input_ids=ids, offset_mapping=[m.span() for m in matches])
    def encode(self, text, **kwargs):
        return self(text)['input_ids']
    def decode(self, ids):
        inverse = {v: k for k, v in self.vocab.items()}
        return ' '.join(inverse[i] for i in ids)


class ToyModel:
    """Residual addition with strongly mediated paths through each head."""
    cfg = SimpleNamespace(n_layers=3, n_heads=2)
    def run_with_hooks(self, tokens, fwd_hooks=()):
        return self.run_with_cache(tokens, fwd_hooks=fwd_hooks)[0]
    def parameters(self):
        return iter([torch.tensor(0.)])
    def run_with_cache(self, tokens, fwd_hooks=(), names_filter=None):
        cache = {}
        def run(name, value):
            for hook_name, callback in fwd_hooks:
                if name == hook_name:
                    value = callback(value, hook=SimpleNamespace(name=name))
            cache[name] = value.clone()
            return value
        resid = tokens.float().unsqueeze(-1)
        for layer in range(3):
            q = run(act_name(layer, 'q'), resid.unsqueeze(2).expand(-1, -1, 2, -1).clone())
            run(act_name(layer, 'k'), q.clone())
            run(act_name(layer, 'v'), q.clone())
            pattern=torch.eye(tokens.shape[1]).expand(tokens.shape[0],2,-1,-1).clone()
            run(act_name(layer,'pattern'),pattern)
            z = run(act_name(layer, 'z'), q * (layer + 1))
            resid = resid + z.sum(2)
            mlp = run(f'blocks.{layer}.hook_mlp_out', torch.zeros_like(resid))
            resid = resid + mlp
        return resid, cache


class DataTests(unittest.TestCase):
    def test_templates_equal_reference_without_importing_legacy_dependencies(self):
        path = Path(__file__).resolve().parents[1] / 'Easy-Transformer/easy_transformer/ioi_dataset.py'
        tree = ast.parse(path.read_text())
        original = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'BABA_TEMPLATES' for t in n.targets))
        self.assertEqual(original, BABA_TEMPLATES)

    def test_positions_corruption_and_answer_exclusion(self):
        tok = WordTokenizer()
        for bos in (False, True):
            for corruption in ('abc', 'subject_swap'):
                examples = build_examples(tok, 17, 42, corruption, bos)
                for ex in examples:
                    positions = ex['positions']
                    self.assertEqual(positions['END'], len(ex['clean_ids']) - 1)
                    self.assertEqual(ex['clean_ids'][positions['IO']], ex['io_id'])
                    self.assertEqual(ex['clean_ids'][positions['S1']], ex['s_id'])
                    self.assertEqual(ex['clean_ids'][positions['S2']], ex['s_id'])
                    self.assertNotIn(ex['clean_ids'][-1], (ex['io_id'], ex['s_id']))
                    changed = [i for i, (a,b) in enumerate(zip(ex['clean_ids'], ex['corrupt_ids'])) if a != b]
                    expected = sorted(positions[r] for r in (('IO','S1','S2') if corruption == 'abc' else ('S2',)))
                    self.assertEqual(changed, expected)
                    if corruption == 'abc':
                        donor_names = [ex['corrupt_ids'][positions[r]] for r in ('IO','S1','S2')]
                        self.assertEqual(len(set(donor_names)), 3)
                        self.assertTrue(set(donor_names).isdisjoint((ex['io_id'], ex['s_id'])))

    def test_pairing_determinism_and_heldout_exclusion(self):
        tok = WordTokenizer()
        train = build_examples(tok, 20, 42)
        self.assertEqual(train, build_examples(tok, 20, 42))
        prompts = {ex['prompt'] for ex in train}
        heldout = build_examples(tok, 20, 43, exclude_prompts=prompts)
        self.assertTrue(prompts.isdisjoint(ex['prompt'] for ex in heldout))
        for first, second in zip(train[::2], train[1::2]):
            self.assertEqual((first['A'],first['B'],first['place'],first['object']),
                             (second['A'],second['B'],second['place'],second['object']))
        for batch in batches(train, 4):
            self.assertLessEqual(len(batch), 4)
            self.assertEqual(len(set(len(ex['clean_ids']) for ex in batch)), 1)


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.model = ToyModel()
        self.clean = torch.tensor([[2,3], [4,5]])
        self.corrupt = torch.tensor([[7,8], [9,10]])
        self.examples = [dict(positions={'END':i, 'S2':1-i}) for i in (0,1)]
        self.clogits, self.cc = self.model.run_with_cache(self.clean)
        self.blogits, self.bc = self.model.run_with_cache(self.corrupt)

    def test_patch_uses_per_example_position_and_does_not_mutate_cache(self):
        original = self.cc[act_name(0,'z')].clone()
        patched = patch_head(self.bc[act_name(0,'z')], self.examples, 1, 'END')(original, None)
        self.assertEqual(patched[0,0,1,0], self.bc[act_name(0,'z')][0,0,1,0])
        self.assertEqual(patched[1,1,1,0], self.bc[act_name(0,'z')][1,1,1,0])
        torch.testing.assert_close(original, self.cc[act_name(0,'z')])
        torch.testing.assert_close(patched[:,:,0], original[:,:,0])

    def test_identical_donor_is_identity_for_paths(self):
        for receiver in (None, Node(2,0,'END'), [Node(1,0,'END'),Node(2,1,'END')]):
            output,_ = path_patch(self.model,self.clean,self.examples,self.cc,self.cc,
                                  Node(0,0,'END'),receiver,freeze_mlps=True)
            torch.testing.assert_close(output,self.clogits)

    def test_path_freezes_intermediates_and_replays_only_receiver(self):
        output,_ = path_patch(self.model,self.clean,self.examples,self.cc,self.bc,
                              Node(0,0,'END'),Node(2,0,'END'),freeze_mlps=True)
        # Sender z delta = 5; direct residual reaches layer2 while layer1
        # output is frozen. Receiver z multiplies by 3 => output delta = 15.
        expected = self.clogits.clone()
        expected[0,0] += 15
        expected[1,1] += 15
        torch.testing.assert_close(output,expected)
        # Ordinary activation patching transmits through intermediate heads.
        ordinary = self.model.run_with_hooks(self.clean,
            [(act_name(0,'z'),patch_head(self.bc[act_name(0,'z')],self.examples,0,'END'))])
        self.assertGreater((ordinary-self.clogits).abs().max(),(output-self.clogits).abs().max())

    def test_rejects_noncausal_receiver(self):
        with self.assertRaises(ValueError):
            path_patch(self.model,self.clean,self.examples,self.cc,self.bc,
                       Node(1,0,'END'),Node(0,0,'END'))

    def test_grouped_scan_keeps_paths_to_later_receivers(self):
        examples = [dict(pattern='ABBA',clean_ids=[2,3],corrupt_ids=[7,8],
                         positions={'END':0},io_id=0,s_id=0)]
        receivers = [Node(1,0,'END'),Node(2,0,'END')]
        records = Experiment(self.model,progress=lambda _:None).scan(
            examples,'path',receivers=receivers)
        self.assertEqual({r['node']['layer'] for r in records},{0,1})
        for record in records:
            expected = [r.as_dict() for r in receivers if r.layer > record['node']['layer']]
            self.assertEqual(record['receivers'],expected)

    def test_batched_edge_confirmation_matches_original(self):
        examples=[dict(pattern='ABBA',clean_ids=[2,3],corrupt_ids=[7,8],
                       positions={'END':0,'IO':0,'S1':1},io_id=0,s_id=0)]
        experiment=Experiment(self.model,progress=lambda _:None)
        senders=[Node(0,0,'END'),Node(1,1,'END')]
        receivers=[Node(1,0,'END'),Node(2,1,'END')]
        original=experiment.confirm_edges(examples,senders,receivers,'q')
        optimized=experiment.confirm_edges_batched(examples,senders,receivers,'q')
        for record in optimized:
            record.pop('node')
        self.assertEqual(original,optimized)

    def test_circuit_extremes(self):
        all_nodes = [Node(l,h,'ALL') for l in range(3) for h in range(2)]
        full = self.model.run_with_hooks(self.clean,circuit_hooks(self.model,self.bc,self.examples,all_nodes))
        torch.testing.assert_close(full,self.clogits)
        removed = self.model.run_with_hooks(self.clean,circuit_hooks(self.model,self.bc,self.examples,[],mode='remove'))
        torch.testing.assert_close(removed,self.clogits)
        empty = self.model.run_with_hooks(self.clean,circuit_hooks(self.model,self.bc,self.examples,[]))
        expected = self.clean.float().unsqueeze(-1) + sum(self.bc[act_name(l,'z')].sum(2) for l in range(3))
        torch.testing.assert_close(empty,expected)

    def test_metric_and_signed_normalization(self):
        examples = [dict(pattern='ABBA',positions={'END':0},io_id=1,s_id=2),
                    dict(pattern='BABA',positions={'END':1},io_id=2,s_id=1)]
        logits = torch.tensor([[[0.,5.,2.],[0.,0.,0.]],[[0.,0.,0.],[0.,1.,6.]]])
        self.assertEqual(logit_diffs(logits,examples).tolist(),[3.,5.])
        noising = effect_summary([3.,5.],[0.,0.],[1.,3.],examples)
        self.assertEqual(noising['all']['delta']['mean'],-2.)
        self.assertEqual(noising['all']['normalized_effect'],-.5)
        denoising = effect_summary([3.,5.],[0.,0.],[1.,3.],examples,'denoising')
        self.assertEqual(denoising['all']['normalized_effect'],.5)
        zero_gap = effect_summary([1.,1.],[1.,1.],[2.,2.],examples)
        self.assertIsNone(zero_gap['all']['normalized_effect'])
        record=dict(node=Node(0,0,'END').as_dict(),effects=noising)
        self.assertEqual(select_nodes([record],1,'negative'),[Node(0,0,'END')])
        self.assertEqual(select_nodes([record],1,'positive'),[])


class ReproductionTests(unittest.TestCase):
    def record(self,head,abba,baba,se=.002):
        return dict(node=Node(0,head,'END').as_dict(),effects={
            p:{'delta':dict(n=200,mean=v,standard_error=se)} for p,v in [('ABBA',abba),('BABA',baba)]})

    def test_screen_retains_both_signs_and_pattern_specific_effects_without_quota(self):
        records=[self.record(h,(-1 if h%2 else 1)*.05,0) for h in range(12)]
        records.append(self.record(12,0,0))
        nodes,decisions=select_supported(records,min_effect=.001)
        self.assertEqual(len(nodes),12)
        self.assertNotIn(Node(0,12,'END'),nodes)
        self.assertTrue(any(d['accepted'] and d['pattern']=='ABBA' and d['mean']<0 for d in decisions))

    def test_small_supported_route_is_retained_under_path_threshold(self):
        records=[self.record(0,-.0043,-.0043,se=.00055),self.record(1,0,0,se=.001)]
        self.assertEqual(select_supported(records,min_effect=.001)[0],[Node(0,0,'END')])
        self.assertEqual(select_supported(records,min_effect=.02)[0],[])

    def test_independent_mean_bank_alignment_and_lookup(self):
        model=ToyModel()
        model.tokenizer=WordTokenizer()
        vocab=dict(names=['John','Mary','Lisa','Mark','David','Sarah'],places=['store'],objects=['book'])
        bank,metadata=independent_means(model,vocab,per_template=1,seed=10,batch_size=2,progress=lambda _:None)
        self.assertEqual(metadata['total_examples'],30)
        self.assertEqual(len(bank),30)
        examples=metadata['examples'][:2]
        donor=means_for_batch(bank,examples,'cpu')
        self.assertEqual(donor[act_name(0,'z')].shape[0],2)
        bad=examples[0].copy()
        bad['template_id']=99
        with self.assertRaises(ValueError):
            means_for_batch(bank,[bad],'cpu')


if __name__ == '__main__':
    unittest.main()
