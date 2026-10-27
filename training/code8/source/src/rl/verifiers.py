"""Task verifiers: ``V(y_hat, y_star) -> accuracy in [0, 1]``.

- ``MathVerifier``: exact match of normalized extracted answers (GSM8K / MATH /
  Big-Math / Reasoning Gym numeric or short-string answers).
- ``CodeVerifier``: fraction of unit tests passed by the decoded program, executed in
  an isolated subprocess with a wall-clock timeout (MBPP ``assert`` lists; APPS
  stdin/stdout cases). Any crash / timeout scores the failing test 0 rather than
  raising, so a bad generation never takes down training.

The code path runs model-generated programs in a subprocess. It is intended for
trusted research datasets on the cluster; harden the sandbox before untrusted input.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from src.rl.answer_extraction import (
    extract_code,
    extract_math_answer,
    gold_from_reference,
    math_equivalent,
)


class Verifier(Protocol):
    def __call__(self, y_hat: str, y_star: object) -> float:
        ...


@dataclass
class MathVerifier:
    """Prediction: marker-based extraction with a last-number fallback (model output is
    unstructured). Reference: trusted as-is (marker if present, else the whole string) -- the
    fallback must never run on the gold, or ``3/4`` becomes ``4``. Marked predictions retain
    complete expressions too. Comparison preserves tuple/interval structure and supports
    bounded symbolic equivalence (``\\sqrt{4}`` == ``2``); see docs/math_verifier.md."""

    def __call__(self, y_hat: str, y_star: object) -> float:
        pred = extract_math_answer(y_hat)
        gold = gold_from_reference(y_star)
        return 1.0 if pred is not None and math_equivalent(pred, gold) else 0.0


def _run_snippet(source: str, stdin: str | None, timeout: float) -> tuple[bool, str]:
    """Run ``source`` in a subprocess; return (ok, stdout)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snippet.py"
        path.write_text(source)
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, ""
        except Exception:
            return False, ""
        return proc.returncode == 0, proc.stdout


@dataclass
class CodeVerifier:
    """y_star is a dict: {"asserts": [...]} OR {"io": [(stdin, expected_stdout), ...]}."""

    timeout: float = 5.0

    def __call__(self, y_hat: str, y_star: object) -> float:
        program = extract_code(y_hat)
        spec = y_star if isinstance(y_star, dict) else {}
        asserts = spec.get("asserts")
        io_cases = spec.get("io")
        if asserts:
            return self._run_asserts(program, list(asserts))
        if io_cases:
            return self._run_io(program, list(io_cases))
        return 0.0

    def _run_asserts(self, program: str, asserts: list[str]) -> float:
        if not asserts:
            return 0.0
        passed = 0
        for check in asserts:
            source = program + "\n" + textwrap.dedent(str(check)) + "\n"
            ok, _ = _run_snippet(source, stdin=None, timeout=self.timeout)
            passed += int(ok)
        return passed / len(asserts)

    def _run_io(self, program: str, io_cases: list) -> float:
        if not io_cases:
            return 0.0
        passed = 0
        for stdin, expected in io_cases:
            ok, out = _run_snippet(program, stdin=str(stdin), timeout=self.timeout)
            if ok and out.strip() == str(expected).strip():
                passed += 1
        return passed / len(io_cases)
