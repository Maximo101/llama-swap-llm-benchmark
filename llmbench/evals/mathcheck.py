"""Final-answer extraction and equivalence checking for the math datasets (GSM8K, MATH-500).

No dependencies. The checker is deliberately conservative: two answers match when their normalised strings are
equal, or when both are plain numbers (integers, decimals, fractions, \\frac{a}{b}) that agree to a 1e-6 relative
tolerance. Anything symbolic that only matches after algebra (x^2+1 vs 1+x^2) counts as wrong, so scores are a
slight LOWER bound on true accuracy rather than an over-estimate.
"""
from __future__ import annotations

import re

from .answers import strip_think

_ANSWER_LINE = re.compile(r"(?i)\banswer\s*(?:is\s*)?[:：]\s*(.+)")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def last_boxed(text):
    """Contents of the LAST \\boxed{...} / \\fbox{...} in `text` (brace-balanced), or None."""
    best = None
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            best = text[m.end():i - 1]
    return best


def _clean_answer_line(s):
    s = s.strip().strip("*_ ").strip()
    boxed = last_boxed(s)
    if boxed is not None:
        return boxed.strip()
    s = s.strip("$ ").strip()
    s = re.sub(r"^\\\(|\\\)$", "", s).strip()
    return s.rstrip(".").strip() or None


def extract_math_answer(text, numeric_fallback=False):
    """The model's final answer: the last 'Answer: ...' line, else the last \\boxed{}, else (GSM8K only, and
    never for a reply that was cut off) the last number in the reply. None when nothing can be found."""
    text = strip_think(text)
    if not text:
        return None
    hits = _ANSWER_LINE.findall(text)
    if hits:
        got = _clean_answer_line(hits[-1])
        if got:
            return got
    boxed = last_boxed(text)
    if boxed is not None and boxed.strip():
        return boxed.strip()
    if numeric_fallback:
        nums = _NUMBER.findall(text)
        if nums:
            return nums[-1]
    return None


def normalize(s):
    """Canonical form of an answer string for comparison."""
    s = str(s).strip()
    boxed = last_boxed(s)
    if boxed is not None:
        s = boxed
    s = s.strip("$ ").strip()
    for a in ("\\!", "\\,", "\\;", "\\:", "\\ ", "\\left", "\\right", "\\displaystyle", "^\\circ", "^{\\circ}",
              "\\%", "%", "\\$", "°"):
        s = s.replace(a, "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = re.sub(r"\\(?:text|textbf|mathrm|mbox|mathbf)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    # trailing unit words: "5 cm", "12 square units" (needs a digit first and >= 2 letters, so "3 x" is kept)
    m = re.fullmatch(r"(-?\d[\d,]*\.?\d*)\s+[A-Za-z]{2,}[A-Za-z .]*", s)
    if m:
        s = m.group(1)
    s = s.replace(" ", "")
    if s.count("=") == 1 and re.match(r"^[A-Za-z]\w*=", s):          # "x=5" -> "5"
        s = s.split("=", 1)[1]
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)                # \frac12 -> \frac{1}{2}
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):               # 1,000 -> 1000
        s = s.replace(",", "")
    m = re.fullmatch(r"\(?([A-Za-z])\)?", s)                         # (B) -> B
    if m:
        s = m.group(1)
    return s.rstrip(".")


def to_number(s):
    """float for plain integers/decimals, 'a/b' and \\frac{a}{b} (optionally negative); None otherwise."""
    s = s.lstrip("+")
    sign = -1.0 if s.startswith("-") else 1.0
    body = s.lstrip("-")
    m = re.fullmatch(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", body) or \
        re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", body)
    try:
        if m:
            den = float(m.group(2))
            return None if den == 0 else sign * float(m.group(1)) / den
        if re.fullmatch(r"\d+(?:\.\d+)?", body):
            return sign * float(body)
    except ValueError:
        pass
    return None


def math_equal(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if p == g:
        return True
    pn, gn = to_number(p), to_number(g)
    if pn is not None and gn is not None:
        return abs(pn - gn) <= 1e-6 * max(1.0, abs(gn))
    return False
