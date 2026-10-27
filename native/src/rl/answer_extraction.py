"""Answer extraction from decoded text for the RL verifiers.

Math answers are pulled from the conventional markers used by the datasets:
GSM8K's ``#### <answer>`` and MATH's ``\\boxed{<answer>}``; otherwise the last
number in the string is used as a fallback. Code "answers" are the decoded program
text (optionally unwrapped from a Markdown code fence).
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from time import monotonic

from latex2sympy2_extended import latex2sympy
from latex2sympy2_extended.latex2sympy2 import ConversionConfig, NormalizationConfig
from math_verify import verify
from math_verify.errors import TimeoutException
from math_verify.utils import timeout

_NUMBER = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_SCALAR = rf"(?:\\(?:d?frac)\{{[^{{}}]+\}}\{{[^{{}}]+\}}|{_NUMBER}(?:\s*/\s*{_NUMBER})?)"
_NUMBER_RE = re.compile(_SCALAR)
_THOUSANDS_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d*)?(?:[eE][-+]?\d+)?")
_ANSWER_MARKER = re.compile(
    r"####|\b(?:final\s+)?answer\s*(?:is\b\s*:?|:)", re.IGNORECASE
)
_UNITS = re.compile(
    r"(?:square\s+|cubic\s+)?(?:dollars?|cents?|minutes?|seconds?|hours?|days?|"
    r"weeks?|months?|years?|feet|foot|inches|inch|meters?|metres?|miles?|kilometers?|"
    r"kilometres?|cm|mm|km|m|kg|grams?|pounds?|liters?|litres?|percent|people|"
    r"students?|apples?|items?|tickets?|books?|pieces?|times)(?:\s+(?:per|a)\s+\w+)?[.!]?",
    re.IGNORECASE,
)


def _clean_prediction_candidate(candidate: str) -> str | None:
    """Strip presentation/known scalar units, never truncate a mathematical expression."""
    candidate = candidate.strip().strip("` ")
    if not candidate:
        return None
    candidate = candidate.splitlines()[0].strip().rstrip(".")
    scalar = re.fullmatch(rf"\$?\s*({_SCALAR})\s*\$?\s*(.*)", candidate)
    if scalar and (not scalar[2] or _UNITS.fullmatch(scalar[2])):
        return scalar[1]
    # Explicit currency in a final-answer sentence, e.g. '$90 if she drives for 10 hours'.
    # Do not use the final number (the duration) as the monetary answer.
    currency = list(re.finditer(rf"\$\s*({_SCALAR})(?![\d.])", candidate))
    if len(currency) == 1 and re.search(r"\b(?:makes?|costs?|earns?|pays?|paid|total)\b", candidate):
        return currency[0][1]
    return candidate


def _last_number(text: str) -> str | None:
    matches = _NUMBER_RE.findall(text)
    return matches[-1] if matches else None


def _extract_boxed(text: str) -> str | None:
    """Extract the content of the last ``\\boxed{...}`` with brace matching."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    brace = text.find("{", idx)
    if brace == -1:
        return None
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : i].strip()
    return None


def extract_math_answer(text: str) -> str | None:
    """Last explicit answer wins; preserve fractions, decimals and structured expressions."""
    text = text.replace("**", "").replace("__", "")
    markers = list(_ANSWER_MARKER.finditer(text))
    marker = markers[-1] if markers else None
    boxed = _extract_boxed(text)
    if boxed is not None and (marker is None or text.rfind("\\boxed") > marker.start()):
        return boxed.strip()
    if marker:
        return _clean_prediction_candidate(text[marker.end():])
    candidate = text.strip().strip("$").strip()
    if candidate and re.fullmatch(r"[A-Za-z\d.,+*/^(){}\[\]\\_\-]+", candidate):
        return candidate
    return _last_number(text)


def gold_from_reference(y_star: object) -> str:
    """The gold answer from a dataset reference -- WITHOUT the last-number fallback.

    A reference is trusted: if it carries an explicit marker (``\\boxed{}`` / ``#### ``) we take
    the marked part, otherwise the whole string IS the answer. Running the prediction-side
    extractor on references turned ``3/4`` (and ``\\frac{3}{4}``) into ``4`` via the last-number
    fallback, marking a correct ``\\boxed{3/4}`` wrong and a wrong ``\\boxed{4}`` right.
    """
    text = str(y_star)
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed.strip()
    if "####" in text:
        return text.split("####")[-1].strip()
    return text.strip()


def _strip_latex_wrappers(a: str) -> str:
    """``\\text{x}`` / ``\\mathrm{x}`` / ``\\textbf{x}`` -> ``x`` (brace-matched, repeated)."""
    for cmd in ("\\text", "\\mathrm", "\\textbf", "\\mathbf", "\\operatorname"):
        while True:
            i = a.find(cmd + "{")
            if i == -1:
                break
            j = i + len(cmd)
            depth, k = 0, j
            while k < len(a):
                if a[k] == "{":
                    depth += 1
                elif a[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            a = a[:i] + a[j + 1 : k] + a[k + 1 :]
    return a


_FRAC_RE = re.compile(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}")
_SQRT_RE = re.compile(r"\\sqrt\{([^{}]*)\}")


def normalize_math_answer(answer: str | None) -> str | None:
    """LaTeX-aware normalization to a canonical plain string.

    ``\\frac{3}{4}`` -> ``3/4``, ``\\sqrt{2}`` -> ``sqrt(2)``, drops ``$``, spacing, ``\\left``/
    ``\\right``, degree/percent marks and ``\\text{}`` wrappers; plain numbers are canonicalised
    (``4.0`` -> ``4``). Use ``math_equivalent`` for the actual comparison.
    """
    if answer is None:
        return None
    a = answer.strip().strip("$").strip()
    a = a.rstrip(".")
    a = _strip_latex_wrappers(a)
    for tok in (
        "\\left",
        "\\right",
        "\\!",
        "\\,",
        "\\;",
        "\\:",
        "\\ ",
        "\\%",
        "%",
        "^{\\circ}",
        "^\\circ",
        "°",
    ):
        a = a.replace(tok, "")
    a = a.replace("\\cdot", "*").replace("\\times", "*")
    for _ in range(3):  # nested fractions
        a = _FRAC_RE.sub(r"(\1)/(\2)", a)
    a = _SQRT_RE.sub(r"sqrt(\1)", a)
    # A comma inside a tuple/set/interval is structural, NOT a thousands separator.
    # Strip numeric grouping only when the entire answer is a grouped scalar.
    if _THOUSANDS_RE.fullmatch(a):
        a = a.replace(",", "")
    a = a.replace(" ", "").replace(r"\{", "{").replace(r"\}", "}")
    # drop trivial parentheses around bare numbers: (3)/(4) -> 3/4
    a = re.sub(r"(?<![\w)])\((-?\d+(?:\.\d+)?)\)", r"\1", a)
    try:
        value = Decimal(a)
        if not value.is_finite():
            return a.lower()
        # Never round distinct values to six significant digits during normalization.
        return (
            str(value.quantize(Decimal(1)))
            if value == value.to_integral()
            else str(value.normalize())
        )
    except InvalidOperation:
        # Symbol names are case-sensitive; ordinary word labels retain legacy casing.
        return a if len(a) == 1 or re.search(r"[+*/^=(),{}\[\]\\-]", a) else a.lower()


_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_FRAC_PLAIN_RE = re.compile(r"^(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)$")


def _to_number(a: str) -> float | None:
    if _NUM_RE.match(a):
        return float(a)
    m = _FRAC_PLAIN_RE.match(a)
    if m:
        den = float(m.group(2))
        return float(m.group(1)) / den if den != 0 else None
    return None


def _collection(answer: str) -> tuple[str, list[str]] | None:
    """Split only top-level delimiters; retain order and open/closed boundaries."""
    if len(answer) < 2 or answer[0] + answer[-1] not in ("()", "[]", "[)", "(]", "{}"):
        return None
    depth, begin, parts = 0, 1, []
    for i, char in enumerate(answer[1:-1], 1):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(answer[begin:i])
            begin = i + 1
        if depth < 0:
            return None
    if depth or (not parts and answer[0] != "{"):
        return None
    parts.append(answer[begin:-1])
    if parts == [""] and answer[0] == "{":
        parts = []
    return answer[0] + answer[-1], parts


@lru_cache(maxsize=2048)
def _parse_scalar(answer: str):
    # Parse the WHOLE extracted expression as math. No numeric/text fallback here:
    # a failed parse must not turn an expression into its final integer.
    try:
        # Math-Verify's default parser lowercases variables. Keep X distinct from x.
        formula = answer.strip().strip("$")
        for _ in range(8):
            formula = re.sub(r"(?<![A-Za-z\\])sqrt\(([^()]*)\)", r"\\sqrt{\1}", formula)
        expr = timeout(2)(latex2sympy)(
            formula,
            normalization_config=NormalizationConfig(
                basic_latex=True,
                units=False,
                malformed_operators=True,
                nits=True,
                boxed="all",
                equations=False,
            ),
            conversion_config=ConversionConfig(lowercase_symbols=False),
        )
        return [expr]
    except (Exception, TimeoutException):  # noqa: BLE001 - third-party parser errors fail closed
        return []


def math_equivalent(
    pred: str | None,
    gold: str | None,
    rel_tol: float = 1e-6,
    *,
    _depth: int = 0,
    _deadline: float | None = None,
) -> bool:
    """Structure-aware comparison shared by RL rewards and evaluation.

    Numeric scalar compatibility is retained; symbolic comparison uses pinned Math-Verify
    with strict variable matching and bounded parsing/comparison. This deliberately does
    not claim bit-for-bit parity with the original MATH string-only grader.
    """
    deadline = monotonic() + 5 if _deadline is None else _deadline
    if (
        pred is None
        or gold is None
        or _depth > 8
        or max(len(pred), len(gold)) > 1024
        or monotonic() >= deadline
    ):
        return False
    p, g = normalize_math_answer(pred), normalize_math_answer(gold)
    if not p or not g or p in ("nan", "inf", "-inf", "zoo") or g in ("nan", "inf", "-inf", "zoo"):
        return False
    pc, gc = _collection(p), _collection(g)
    if pc is not None or gc is not None:
        if pc is None or gc is None or pc[0] != gc[0]:
            return False
        if max(len(pc[1]), len(gc[1])) > 32:
            return False

        def same(a, b):
            return math_equivalent(a, b, rel_tol, _depth=_depth + 1, _deadline=deadline)

        if pc[0] == "{}":
            # Finite sets are unordered; duplicates do not change a mathematical set.
            return all(any(same(a, b) for b in gc[1]) for a in pc[1]) and all(
                any(same(a, b) for a in pc[1]) for b in gc[1]
            )
        return len(pc[1]) == len(gc[1]) and all(same(a, b) for a, b in zip(pc[1], gc[1]))
    if re.fullmatch(r"[-+]?\d+", p) and re.fullmatch(r"[-+]?\d+", g):
        return int(p) == int(g)
    pn, gn = _to_number(p), _to_number(g)
    if pn is not None and gn is not None:
        return abs(pn - gn) <= rel_tol * max(1.0, abs(gn))
    # Non-mathematical short strings preserve their historical normalized comparison.
    # Do not send literal unit-bearing labels through expression extraction.
    if p == g and not re.search(r"[\\+*/^=\-]", p + g):
        return True
    parsed_gold, parsed_pred = _parse_scalar(gold), _parse_scalar(pred)
    return bool(
        parsed_gold
        and parsed_pred
        and verify(parsed_gold, parsed_pred, strict=True, timeout_seconds=2)
    )


def extract_code(text: str) -> str:
    """Unwrap a Markdown ```python ...``` fence if present, else return text."""
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()
