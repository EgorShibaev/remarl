"""Official LLaDA forward sharing the existing frozen checkpoint tensors."""
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

BASE = Path(__file__).resolve().parent


def reference(engine):
    sys.path.insert(0, str(BASE/'sources/WINO/LLaDA'))
    import configuration_llada as configuration
    import modeling_llada as modeling
    c = engine.config
    config = configuration.ModelConfig(
        d_model=c.d_model, n_heads=c.n_heads, n_kv_heads=c.n_kv_heads,
        n_layers=c.n_layers, mlp_hidden_size=c.mlp_hidden_size,
        vocab_size=c.embedding_size, embedding_size=c.embedding_size,
        max_sequence_length=max(c.max_sequence_length, engine.sin.shape[-2]),
        rope=True, rope_theta=c.rope_theta, rope_full_precision=True,
        rms_norm_eps=c.rms_norm_eps, layer_norm_type='rms', block_type='llama',
        activation_type='silu', weight_tying=False, include_bias=False,
        include_qkv_bias=False, layer_norm_with_affine=True, bias_for_layer_norm=False,
        attention_dropout=0., embedding_dropout=0., residual_dropout=0.,
        flash_attention=False, init_device='meta')
    model = modeling.LLaDAModel(config, init_params=False)
    model.load_state_dict({k.removeprefix('model.'): v.detach()
                           for k, v in engine.weight_mapping().items()}, strict=True, assign=True)
    model.eval().requires_grad_(False)
    return model


class Adapter:
    def __init__(self, model):
        self.model = model
        self.transformer = model.transformer
        self.config = SimpleNamespace(num_hidden_layers=model.config.n_layers)
        self.device = next(model.parameters()).device
        self.calls = 0

    @torch.inference_mode()
    def __call__(self, ids, attention_mask=None, **kwargs):
        # All evaluation prompts have batch size one and no padding. The WINO
        # model's mask API is 4-D; an all-ones 2-D mask means unrestricted SDPA.
        if attention_mask is not None:
            assert attention_mask.shape == ids.shape and attention_mask.bool().all()
        self.calls += 1
        return self.model(ids, **kwargs)
