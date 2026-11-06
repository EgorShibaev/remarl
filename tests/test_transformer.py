"""Architecture and answer-boundary checks, independent of the frozen model."""
import unittest
from types import SimpleNamespace

import torch

from remarl.policy import ReMaRLPolicy as SetSelector, PolicyConfig as SetSelectorConfig
from remarl.transformer_policy import TransformerPolicy as TransformerSelector, transformer_config
from remarl.optimization import replay_error
from remarl.rollout import collect
from remarl.runtime import canvas


def config():
    return SetSelectorConfig(input_node_dim=3, hidden_dim=64, architecture="mlp",
        linear_skip=True, position_encoding="learned", position_column=2,
        position_count=256, position_dim=16, feature_encoding="none",
        backbone_hidden_dim=4096, hidden_projection_dim=64, dropout=0.0)


def features(batch=2):
    f = torch.randn(batch, 256, 4099)
    f[..., 0] = torch.rand(batch, 256)
    f[..., 1] = torch.randint(2, (batch, 256)).float()
    f[..., 2] = torch.arange(256)
    return f


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.model = TransformerSelector(transformer_config(config())).eval()

    def test_exact_initial_confidence_and_shared_features(self):
        f = features()
        self.assertTrue(torch.equal(self.model(f, f[..., 0]), f[..., 0]))
        torch.manual_seed(7)
        old = SetSelector(config())
        for name, value in old.state_dict().items():
            if name.startswith(("hidden_projection.", "position_embedding.", "linear_skip.")):
                self.assertTrue(torch.equal(value, self.model.state_dict()[name]), name)
        self.assertEqual(sum(p.numel() for p in old.parameters()), 284115)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 583187)

    def test_no_attention_between_answers(self):
        torch.nn.init.normal_(self.model.output_projection.weight, std=0.1)
        f = features()
        with torch.no_grad():
            together = self.model(f, f[..., 0])
            separate = torch.cat([self.model(v[None], v[None, :, 0]) for v in f])
            torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-6)
            altered = f.clone(); altered[1, :, 3:] = 100 * torch.randn_like(altered[1, :, 3:])
            again = self.model(altered, altered[..., 0])
            self.assertTrue(torch.equal(together[0], again[0]))
            self.assertFalse(torch.equal(together[1], again[1]))

    def test_attention_can_use_other_positions(self):
        torch.nn.init.normal_(self.model.output_projection.weight, std=0.1)
        f = features(1)
        before = self.model(f, f[..., 0])
        changed = f.clone(); changed[0, 180, 3:] = torch.randn_like(changed[0, 180, 3:]) * 10
        after = self.model(changed, changed[..., 0])
        self.assertGreater(float((before[0, 0] - after[0, 0]).detach().abs()), 1e-8)

    def test_refuses_flattened_or_reordered_answers(self):
        f = features()
        with self.assertRaises(ValueError):
            self.model(f.flatten(0, 1), f[..., 0].flatten())
        with self.assertRaises(ValueError):
            self.model(f[:, :128], f[:, :128, 0])
        with self.assertRaises(ValueError):
            self.model(f.flip(1), f.flip(1)[..., 0])

    def test_gradients_reach_attention_and_projection(self):
        f = features(1).requires_grad_(True)
        target = torch.randn(1, 256)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0)
        for _ in range(2):
            optimizer.zero_grad()
            (self.model(f, f[..., 0]) * target).mean().backward()
            optimizer.step()
        for name in ("transformer.q_proj.weight", "transformer.v_proj.weight",
                     "hidden_projection.1.weight", "position_embedding.weight"):
            grad = dict(self.model.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum() > 0, name)
        self.assertEqual(float(f.grad[..., 3:].abs().sum()), 0.0)

    def test_rollout_likelihoods_replay_with_answer_boundaries(self):
        torch.nn.init.normal_(self.model.output_projection.weight, std=.01)

        def backbone(ids, attention, positions):
            answer = ids.gather(1, positions)
            logits = torch.arange(17).float().expand(*answer.shape, 17)
            hidden = answer[..., None].float().expand(*answer.shape, 4096) / 1000
            return logits, hidden

        ids, attention, positions = canvas([dict(prompt_ids=[2, 3])] * 8, 0, 256)
        rollout = collect(SimpleNamespace(mask_token_id=126336), self.model,
            ids, attention, positions, torch.rand(8, 8, 256),
            answer_backbone=backbone, record=True)
        groups = [dict(question_id=0, rewards=[0, 1] * 4, replay=rollout)]
        self.assertLess(replay_error(self.model, groups, "cpu"), .003)


if __name__ == "__main__":
    unittest.main()
