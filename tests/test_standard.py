"""Small state-dependent reference comparisons; no model downloads or training."""

from types import SimpleNamespace
import torch
from remarl.baselines.standard import decode, definitions, UPSTREAM


class Dummy:
    device = torch.device("cpu")
    dtype = torch.float32
    config = SimpleNamespace(vocab_size=17)

    def __init__(self, tied=False):
        self.calls = 0
        self.trace = []
        self.tied = tied

    def __call__(self, x, **kwargs):
        self.calls += 1
        self.trace.append(x.clone())
        positions = torch.arange(x.shape[1]).float()[None, :, None]
        vocabulary = torch.arange(17).float()[None, None]
        # Changes when visibility and tokens change; ties have a separate test.
        phase = float(x.remainder(17).sum()) * 0.17
        logits = torch.sin(positions * 0.41 + vocabulary * 1.31 + phase) * (1 + positions * 0.11)
        if self.tied:
            logits = torch.ones_like(logits)
        return SimpleNamespace(logits=logits)


def run():
    torch.set_num_threads(1)
    source = UPSTREAM / "Uncode/src/generate.py"
    names = [
        "generate_with_pc_sampler",
        "generate_with_eb_sampler",
        "pc_sampler_function",
        "get_num_transfer_tokens",
        "add_gumbel_noise",
    ]
    freq = torch.linspace(0.001, 0.05, 17)
    unc = definitions(source, names, {"BASE_LINE": freq})
    klass = definitions(
        UPSTREAM / "KLASS/src/model/llada_klass.py",
        ["stable_confident_decode", "get_num_transfer_tokens", "add_gumbel_noise"],
    )
    prompt = torch.tensor([[2, 3, 4]])
    checked = []
    for steps in (2, 4, 8, 16):
        for block in (8, 16):
            cfg = dict(mode="pc_sampler", block=block, steps=steps)
            a, b = Dummy(), Dummy()
            result = decode(a, prompt, cfg, length=16, frequency=freq)
            reference = unc["generate_with_pc_sampler"](
                b,
                prompt,
                steps=steps,
                gen_length=16,
                block_length=block,
                lambd=0.25,
                alpha=10.0,
                temperature=0.0,
            )
            assert torch.equal(result.sequences, reference), cfg
            assert a.calls == b.calls == steps
            assert all(torch.equal(x, y) for x, y in zip(a.trace, b.trace))
            checked.append(cfg)
    for gamma in (0.001, 0.1, 1.0, 10.0):
        cfg = dict(mode="eb_sampler", block=16, gamma=gamma)
        a, b = Dummy(), Dummy()
        result = decode(a, prompt, cfg, length=16)
        reference = unc["generate_with_eb_sampler"](b, prompt, gamma=gamma, gen_length=16)
        assert torch.equal(result.sequences, reference), cfg
        assert a.calls == b.calls
        assert all(torch.equal(x, y) for x, y in zip(a.trace, b.trace))
        checked.append(cfg)
    for block in (4, 16):
        for kl in (0.001, 0.01, 0.015, 0.1, 100.0):
            cfg = dict(mode="klass", block=block, kl=kl, confidence=0.1)
            a, b = Dummy(), Dummy()
            result = decode(a, prompt, cfg, length=16)
            reference, calls = klass["stable_confident_decode"](
                b,
                None,
                prompt,
                gen_length=16,
                steps=16,
                block_length=block,
                conf_threshold=0.1,
                kl_threshold=kl,
                alg="klass",
                unmask_strategy="all",
            )
            assert torch.equal(result.sequences, reference), cfg
            assert a.calls == b.calls == calls
            assert all(torch.equal(x, y) for x, y in zip(a.trace, b.trace))
            checked.append(cfg)
    for mode in ("margin", "eb_sampler"):
        for block in (4, 16):
            cfg = dict(mode=mode, block=block, steps=8, gamma=0.1)
            model = Dummy()
            result = decode(model, prompt, cfg, length=16)
            assert torch.equal(result.sequences[:, :3], prompt)
            for x in model.trace:
                answer = x[0, 3:]
                first_mask = answer.eq(126336).nonzero().flatten()
                if first_mask.numel():
                    current_block = int(first_mask[0]) // block
                    assert answer[(current_block + 1) * block :].eq(126336).all()
            checked.append(cfg)
    for tied in (False, True):
        for steps in (1, 2, 4, 8, 16):
            cfg = dict(mode="p2_original", block=16, steps=steps)
            a, b = Dummy(tied), Dummy(tied)
            result = decode(a, prompt, cfg, length=16)
            # Literal Appendix-E loop with a fixed prompt and eta=1/tau=0.
            x = torch.full((1, 19), 126336, dtype=torch.long)
            x[:, :3] = prompt
            fixed = x.ne(126336)
            for step in range(1, steps + 1):
                logits = b(x).logits.double()
                was_masked = x.eq(126336)
                score, proposal = logits.log_softmax(-1).max(-1)
                score.masked_fill_(fixed, float("inf"))
                cutoff = ((~fixed).sum(1, keepdim=True).float() * (1 - step / steps)).long()
                remask = score < score.sort(-1).values.gather(-1, cutoff)
                x[remask] = 126336
                select = was_masked & ~remask
                x[select] = proposal[select]
            remaining = x.eq(126336)
            x[remaining] = proposal[remaining]
            assert torch.equal(result.sequences, x), (tied, steps)
            assert all(torch.equal(x, y) for x, y in zip(a.trace, b.trace))
            checked.append(dict(**cfg, tied=tied))
    return dict(
        passed=True,
        reference_trace_checks=len(checked),
        checks=checked,
        upstream_sources=["UNCODE PC/EB", "KLASS", "P2 Appendix E"],
        block_invariant_checks=True,
        training=False,
    )


def test_standard_reference_traces(upstream_sources):
    assert run()["passed"]
