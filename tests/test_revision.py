"""Reference-model logits, full decoder traces, posterior and NFE tests."""

import importlib.util
import sys
from types import SimpleNamespace

import torch
from remarl.baselines.adapters import (
    ExplicitAttentionAdapter,
    definitions,
    remdm_uncode,
    remdm_probabilities,
    wino,
)

from remarl.baselines.sources import source_root

BASE = source_root()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Recording:
    def __init__(self, model):
        self.model = model
        self.device = (
            next(model.parameters()).device if hasattr(model, "parameters") else model.device
        )
        self.calls = 0
        self.trace = []

    def __call__(self, x, **kwargs):
        self.calls += 1
        self.trace.append(x.clone())
        return self.model(x, **kwargs)


def reference(engine, module, config_module):
    c = engine.config
    config = config_module.ModelConfig(
        d_model=c.d_model,
        n_heads=c.n_heads,
        n_kv_heads=c.n_kv_heads,
        n_layers=c.n_layers,
        mlp_hidden_size=c.mlp_hidden_size,
        vocab_size=c.embedding_size,
        embedding_size=c.embedding_size,
        max_sequence_length=c.max_sequence_length,
        rope=True,
        rope_theta=c.rope_theta,
        rope_full_precision=True,
        rms_norm_eps=c.rms_norm_eps,
        layer_norm_type="rms",
        block_type="llama",
        activation_type="silu",
        weight_tying=False,
        include_bias=False,
        include_qkv_bias=False,
        layer_norm_with_affine=True,
        bias_for_layer_norm=False,
        attention_dropout=0.0,
        embedding_dropout=0.0,
        residual_dropout=0.0,
        flash_attention=False,
        init_device="meta",
    )
    model = module.LLaDAModel(config, init_params=False)
    # Aliased views: no second copy of the 8B model on the GPU.
    model.load_state_dict(
        {k.removeprefix("model."): v.detach() for k, v in engine.weight_mapping().items()},
        strict=True,
        assign=True,
    )
    model.eval().requires_grad_(False)
    return model


def run(gpu=False, checkpoint=None):
    torch.set_num_threads(2)
    from remarl import backbone as port

    sys.path.insert(0, str(BASE / "WINO/LLaDA"))
    import configuration_llada as config_module
    import modeling_llada as module

    device = "cuda" if gpu else "cpu"
    cases = []
    if checkpoint:
        engine = port.LLaDAOfflineModel.from_local_checkpoint(
            checkpoint, device=device, backend="torch", fuse_projections=False
        )
    else:
        torch.manual_seed(7)
        c = port.LLaDAConfig(16, 2, 2, 2, 24, 128, 1400)
        engine = port.LLaDAOfflineModel(
            c, dtype=torch.float32, backend="torch", fuse_projections=False
        ).to(device)
        engine.load_weights(
            (k, torch.randn_like(v) * 0.2) for k, v in engine.weight_mapping().items()
        )
    ours = ExplicitAttentionAdapter(engine)
    ref = reference(engine, module, config_module)
    lengths = (16, 35) if checkpoint else (16, 35, 529, 1169)
    with torch.inference_mode():
        for length in lengths:
            ids = torch.randint(
                0, min(engine.config.embedding_size, 100), (1, length), device=device
            )
            positions = torch.arange(length, device=device)
            positions[-4:] = positions[-8:-4]
            mask = torch.ones((1, 1, length, length), dtype=torch.bool, device=device)
            mask[..., -4:] = False
            mask[:, :, -4:, -4:] = True
            mask[:, :, -4:, -8:-4] = ~torch.eye(4, dtype=torch.bool, device=device)
            actual = ours(ids, attention_mask=mask, position_ids=positions).logits
            expected = ref(ids, attention_mask=mask, position_ids=positions).logits
            error = float((actual.float() - expected.float()).abs().max())
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            cases.append(dict(test="official_WINO_logits", length=length, max_error=error))
        # Full trajectories compare against the official model + decoder,
        # not merely a second invocation of our forward implementation.
        prompt = torch.tensor([[2, 3, 4]], device=device)
        mask_id = 126336 if checkpoint else 127
        source = BASE / "WINO/LLaDA/decoding.py"
        decoder = definitions(source, ["decoding_wino", "add_gumbel_noise"])["decoding_wino"]
        for block in (8, 16):
            a, b = Recording(ours), Recording(ref)
            result = wino(a, prompt, source=source, length=16, block=block, mask_id=mask_id)
            expected, calls = decoder(
                b, prompt, gen_length=16, block_length=block, mask_id=mask_id, temperature=0.0
            )
            assert torch.equal(result.sequences, expected)
            assert int(result.steps_taken[0]) == a.calls == b.calls == calls
            assert all(torch.equal(x, y) for x, y in zip(a.trace, b.trace))
            cases.append(dict(test="WINO_full_trajectory", block=block, calls=calls))
        upstream = definitions(
            BASE / "Uncode/src/generate.py", ["generate_with_remdm", "add_gumbel_noise"]
        )["generate_with_remdm"]
        for k in (1, 2):
            a, b = Recording(ours), Recording(ref)
            result = remdm_uncode(
                a, prompt, length=16, initial_fraction=0.875, k=k, loops=3, mask_id=mask_id
            )
            expected = upstream(
                b,
                prompt,
                gen_length=16,
                init_unmask_ratio=0.875,
                unmask_k=k,
                loop_steps=3,
                mask_id=mask_id,
                temperature=0.0,
                tokenizer=SimpleNamespace(decode=lambda *a, **kw: ""),
            )
            assert torch.equal(result.sequences, expected)
            assert a.calls == b.calls == 16 // k + 6
            assert all(torch.equal(x, y) for x, y in zip(a.trace, b.trace))
            cases.append(dict(test="ReMDM_UNCODE_full_trajectory", k=k, calls=a.calls))
        # Compare posterior tensors with the official ReMDM function directly.
        tree = __import__("ast").parse((BASE / "ReMDM/diffusion.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, __import__("ast").ClassDef) and n.name == "Diffusion"
        )
        fn = next(n for n in cls.body if getattr(n, "name", "") == "_ddpm_caching_update")
        captured = []

        def capture(q):
            captured.append(q.clone())
            return q.argmax(-1)

        scope = dict(torch=torch, _sample_categorical=capture)
        exec(
            compile(__import__("ast").Module(body=[fn], type_ignores=[]), "official_ReMDM", "exec"),
            scope,
        )
        for variant in ("cap", "rescale"):
            for at, as_ in ((0.0, 0.25), (0.5, 0.75), (0.875, 1.0)):
                for eta in (0.0, 0.015, 1.0):
                    x = torch.tensor([[1, 4, 2, 4]], device=device)
                    p = torch.softmax(torch.randn(1, 4, 5, device=device), -1)
                    clean = p.clone()
                    clean[..., 4] = 0
                    clean /= clean.sum(-1, keepdim=True)
                    visible = x.ne(4)
                    clean[visible] = torch.nn.functional.one_hot(x[visible], 5).to(p.dtype)
                    fake = SimpleNamespace(
                        config=SimpleNamespace(
                            noise=SimpleNamespace(type="loglinear"),
                            sampling=SimpleNamespace(
                                sampler="remdm-" + variant, eta=eta, nucleus_p=1.0
                            ),
                        ),
                        mask_index=4,
                        noise=lambda t: (-torch.log1p(-t), None),
                        time_conditioning=False,
                    )
                    scope["_ddpm_caching_update"](
                        fake, x, torch.tensor([1 - at], device=device), as_ - at, p_x0=clean
                    )
                    expected = captured[-1]
                    actual = remdm_probabilities(p, x, at, as_, eta, 4, variant)
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
                    torch.testing.assert_close(actual.sum(-1), torch.ones_like(actual[..., 0]))
                    assert (actual >= 0).all()
                    cases.append(
                        dict(test="official_ReMDM_posterior", variant=variant, at=at, eta=eta)
                    )
    return dict(passed=True, gpu=gpu, real_checkpoint=bool(checkpoint), cases=cases)


def test_revision_reference_traces(upstream_sources):
    assert run()["passed"]
