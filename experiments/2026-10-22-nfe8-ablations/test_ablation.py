"""CPU tests, run inside the pinned cluster runtime before any experiment."""
import itertools
import math
import unittest
import torch
import ablation_ops as ops
from src.models.set_selector import SetSelector, SetSelectorConfig

CFG = SetSelectorConfig(input_node_dim=3,architecture='mlp',hidden_dim=64,
    dropout=0.,linear_skip=True,position_encoding='learned',position_column=2,
    backbone_hidden_dim=4096,hidden_projection_dim=64)


class AblationTests(unittest.TestCase):
    def tearDown(self):
        ops.configure('full')

    def features(self):
        x = torch.randn(256,4099)
        x[:,0] = x[:,0].sigmoid(); x[:,1] = torch.arange(256)%2; x[:,2] = torch.arange(256)
        return x

    def model(self,mode):
        ops.configure(mode); torch.manual_seed(7)
        return ops.selector_class(SetSelector)(CFG).eval()

    def test_all_initializations_are_exact_p2(self):
        f = self.features(); scores = f[:,0]
        for mode in ('no_revision','no_hidden','no_position','bias_only','full'):
            model = self.model(mode)
            self.assertTrue(torch.equal(model(f,scores),scores),mode)

    def test_hidden_removal_is_invariant_after_nonzero_head(self):
        model = self.model('no_hidden')
        torch.nn.init.normal_(model.pointwise[-1].weight)
        f = self.features(); changed = f.clone(); changed[:,3:] = torch.randn_like(changed[:,3:])*50
        self.assertTrue(torch.equal(model.delta(f),model.delta(changed)))
        self.assertTrue(all(not p.requires_grad for p in model.hidden_projection.parameters()))

    def test_position_removal_is_invariant_after_nonzero_head(self):
        model = self.model('no_position')
        torch.nn.init.normal_(model.pointwise[-1].weight)
        f = self.features(); changed = f.clone(); changed[:,2] = changed[:,2].flip(0)
        self.assertTrue(torch.equal(model.delta(f),model.delta(changed)))
        self.assertFalse(model.position_embedding.weight.requires_grad)

    def test_bias_only_has_only_requested_information(self):
        model = self.model('bias_only')
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),258)
        torch.nn.init.normal_(model.position_bias); torch.nn.init.normal_(model.mask_bias)
        f = self.features(); changed = f.clone(); changed[:,0] = 1-f[:,0]; changed[:,3:] *= 100
        self.assertTrue(torch.equal(model.delta(f),model.delta(changed)))

    def test_full_pl_matches_reference(self):
        from worker import ordered_logprob
        z = torch.randn(8,256,requires_grad=True); u = torch.rand_like(z)
        ops.configure('full')
        order = ops.select_order(z,u,96,torch.ones_like(z,dtype=torch.bool))
        self.assertTrue(torch.equal(ops.action_logprob(z,order,z>0),ordered_logprob(z,order)))

    def test_monotone_action_support_and_exact_normalization(self):
        ops.configure('no_revision')
        z = torch.tensor([[.2,.7,-.1,1.1]],requires_grad=True)
        mask = torch.tensor([[False,True,True,True]])
        probabilities = []
        for new in itertools.permutations((1,2,3),2):
            order = torch.tensor([[0,*new]])
            lp = ops.action_logprob(z,order,mask)
            probabilities.append(lp.exp())
        self.assertAlmostEqual(float(torch.stack(probabilities).sum()),1.,places=6)
        for _ in range(12):
            order = ops.select_order(z,torch.rand_like(z),3,mask)
            self.assertEqual(order[0,0],0)
            self.assertEqual(len(set(order[0].tolist())),3)
            lp = ops.action_logprob(z,order,mask)
            self.assertTrue(torch.isfinite(lp).all())
            lp.sum().backward();self.assertTrue(torch.isfinite(z.grad).all());z.grad=None

    def test_greedy_and_retention_bonus(self):
        z = torch.tensor([[.1,.8,.3,.5]])
        m = torch.tensor([[False,True,True,True]])
        ops.configure('p2',0.)
        self.assertEqual(ops.select_order(z,torch.rand_like(z),2,m).tolist(),[[1,3]])
        ops.configure('p2',0.,2.)
        self.assertEqual(ops.select_order(z,torch.rand_like(z),2,m).tolist(),[[0,1]])


if __name__=='__main__':
    unittest.main(verbosity=2)
