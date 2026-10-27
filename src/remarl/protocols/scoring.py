"""Shared APPLe math scoring and isolated code execution."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
from .math import official, score

OFFICIAL = Path(__file__).parent / "official"


class Scorer:
    def __init__(self, tokenizer, domain):
        self.tokenizer, self.domain = tokenizer, domain
        self.api = official(OFFICIAL)
        self.pool = ThreadPoolExecutor(max_workers=8)
        if domain == "code":
            from . import sandbox

            sandbox.setup()  # All isolation probes must pass before model code runs.
            self.sandbox = sandbox
            spec = importlib.util.spec_from_file_location("remarl_sanitize", OFFICIAL / "sanitize.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.sanitize = module

    def begin_collection(self):
        if self.domain == "code":
            good = self.sandbox.run("assert 2 + 2 == 4")
            bad = self.sandbox.run("assert False")
            if not good["passed"] or bad["passed"]:
                raise RuntimeError("Code reward health check failed")

    def score_one(self, tokens, gold):
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        if gold["task"] in ("gsm8k", "math"):
            return score(self.api, text, gold)
        if gold["task"] == "mbpp":
            code = self.sanitize.sanitize_mbpp(text)
        else:
            code = self.sanitize.sanitize_humaneval(gold["raw_prompt"] + text, gold["entry_point"])
        verdict = self.sandbox.run(code + "\n" + gold["tests"])
        if verdict["result"] not in ("passed", "timed out") and not verdict["result"].startswith(
            "failed:"
        ):
            raise RuntimeError(f"Sandbox infrastructure failure: {verdict}")
        return dict(
            text=text,
            task=gold["task"],
            reward=float(verdict["passed"]),
            execution_result=verdict["result"],
        )

    def __call__(self, tokens, golds):
        if len(tokens) != len(golds):
            raise ValueError("Tokens/golds length mismatch")
        return list(self.pool.map(self.score_one, tokens, golds))

    def close(self):
        self.pool.shutdown(wait=True)
