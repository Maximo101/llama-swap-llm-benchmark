"""Parsing model replies: reasoning-block stripping, MC letters, code extraction."""
from __future__ import annotations

import ast
import re

def strip_think(text):
    """Remove reasoning blocks so answers are only looked for in the final reply."""
    text = text or ""
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.S | re.I)
    if re.search(r"</think(?:ing)?>", text, re.I):          # the opening tag was consumed by the chat template
        text = re.split(r"</think(?:ing)?>", text, flags=re.I)[-1]
    text = re.sub(r"<think(?:ing)?>.*\Z", "", text, flags=re.S | re.I)   # truncated thinking: no answer
    return text.strip()


_KEYWORD = re.compile(r"(?i:final\s+answer|correct\s+(?:answer|option|choice)|answer|choice|option)")
_BOXED = re.compile(r"\\boxed\{\s*(?:\\text(?:bf)?\{\s*)?\(?\s*([A-J])\b")
_LEAD = re.compile(r"^(?:[\s:=\-\u2192*_`\"'(\[{$\\]+|is\b|are\b|option\b|choice\b|\\?boxed\{|\\?text(?:bf)?\{)", re.I)


def _letter_after(tail, letters):
    """Letter that directly follows an 'Answer:'-style keyword, or None."""
    t = tail[:60]
    while True:
        m = _LEAD.match(t)
        if not m:
            break
        t = t[m.end():]
    m = re.match(r"([A-J])(?![A-Za-z0-9])", t)
    if not m or m.group(1) not in letters:
        return None
    if m.group(1) in "AI" and re.match(r"\s+[a-z]{2,}", t[1:]):    # "answer is A great ...", "Answer: I think"
        return None
    return m.group(1)


def extract_choice(text, letters="ABCDEFGHIJ"):
    """Final multiple-choice letter from a free-form reply, or None. Looks for (in order of trust, taking the
    LAST occurrence): 'Answer: X' / 'the answer is X' / 'correct option is X', \\boxed{X}, then a last line
    that is just the letter."""
    text = strip_think(text)
    if not text:
        return None
    hits = []
    for m in _KEYWORD.finditer(text):
        got = _letter_after(text[m.end():], letters)
        if got:
            hits.append((m.start(), got))
    for m in _BOXED.finditer(text):
        if m.group(1) in letters:
            hits.append((m.start(), m.group(1)))
    if hits:
        return max(hits)[1]
    lines = [l for l in text.splitlines() if l.strip()]
    if lines:
        m = re.match(r"^\W*\(?([A-J])\)?[.):]?\W*$", lines[-1].strip())
        if m and m.group(1) in letters:
            return m.group(1)
    return None


_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[^\n]*\n(.*?)(?:```|\Z)", re.S)
_CODEISH = re.compile(r"\b(def|class|import|from|print|input|return|for|while)\b|sys\.stdin")


def extract_code(text, entry=None):
    """Python source from a reply: the last fenced python block (preferring one that defines `entry`).
    Tolerates a missing closing fence (truncated replies) and replies with no fence at all."""
    text = strip_think(text)
    blocks = [(lang.lower(), body.strip("\n")) for lang, body in _FENCE.findall(text) if body.strip()]
    py = [b for lang, b in blocks if lang in ("", "python", "py", "python3", "py3")] or [b for _, b in blocks]
    if entry:
        hit = [b for b in py if re.search(rf"\bdef\s+{re.escape(entry)}\b", b)]
        if hit:
            return hit[-1]
    real = [b for b in py if _CODEISH.search(b)]
    if real:
        return real[-1]
    if py:
        return py[-1]
    return text


def strip_main_guard(code):
    """Drop top-level `if __name__ == '__main__':` blocks (example usage that may read stdin or crash).
    Used for HumanEval / MBPP only; competitive-programming solutions legitimately run from main()."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    def guard(n):
        return (isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                and n.test.left.id == "__name__")
    if not any(guard(n) for n in tree.body) or not hasattr(ast, "unparse"):
        return code
    tree.body = [n for n in tree.body if not guard(n)]
    return ast.unparse(tree)
