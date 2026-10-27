"""Test official decoder execution, call accounting, prompt and budget invariants."""

from pathlib import Path
import warnings
import torch
from remarl.baselines.extra import decode, core_steps, grid
from remarl.baselines.model_adapter import Adapter, reference

BASE = Path(__file__).resolve().parent


def run(gpu=False, checkpoint=None):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    from remarl import backbone as port

    device = "cuda" if gpu else "cpu"
    if checkpoint:
        engine = port.LLaDAOfflineModel.from_local_checkpoint(
            checkpoint, device=device, backend="torch", fuse_projections=False
        )
    else:
        config = port.LLaDAConfig(16, 2, 2, 2, 24, 128, 5000)
        engine = port.LLaDAOfflineModel(
            config, dtype=torch.float32, backend="torch", fuse_projections=False
        ).to(device)
        engine.load_weights(
            (k, torch.randn_like(v) * 0.2) for k, v in engine.weight_mapping().items()
        )
        # Ensure the toy model never proposes its mask token.
        engine.head.weight.data[127].zero_()
    model = reference(engine)
    mask = 126336 if checkpoint else 127
    prompt = torch.tensor([[2, 3, 4]], device=device)
    tests = []
    for n in [4, 8, 16, 32, 64, 128, 256]:
        s = core_steps(n)
        assert s + sum(i % 8 == 0 and int(0.25 * s) <= i < int(0.75 * s) for i in range(s)) <= n
        tests.append(dict(test="core_reserved_budget", budget=n, base_steps=s))
    configs = [dict(mode="rcr", steps=n, block=64) for n in (4, 8, 16)]
    configs += [dict(mode="core", budget=n, block=64) for n in (8, 16, 32)]
    configs += [
        dict(mode="dapd", block=b, variant=v)
        for b in (16, 64)
        for v in ("dapd_direct", "dapd_staged")
    ]
    for cfg in configs:
        adapter = Adapter(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            out = decode(adapter, prompt, cfg, length=64, mask_id=mask)
        assert out.sequences.shape == (1, 67)
        assert int(out.steps_taken[0]) == adapter.calls
        assert not any(p.requires_grad for p in model.parameters())
        tests.append(dict(test="official_decoder", config=cfg, calls=adapter.calls))
    # Full-length RCR and CoRe reference functions are called verbatim; DAPD's
    # hook must restore the original model methods after every invocation.
    from dapd.core import AttentionCaptureHook

    ids = torch.randint(1, 100, (1, 73), device=device)
    with torch.inference_mode():
        expected = model(ids).logits
        hook = AttentionCaptureHook(capture_layers=[model.config.n_layers - 1])
        hook.register(Adapter(model))
        try:
            actual = model(ids).logits
            attention = hook.compute_attention_weights()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(attention.sum(-1), torch.ones_like(attention.sum(-1)))
        finally:
            hook.restore()
    tests.append(dict(test="DAPD_hook_preserves_logits_and_normalization"))
    assert len(grid("math")) == 36 and len(grid("code")) == 32
    return dict(passed=True, gpu=gpu, checkpoint=bool(checkpoint), tests=tests)


def test_extra_baseline_traces(upstream_sources):
    assert run()["passed"]
