"""v2.3: write .env in the .env.example layout, and favourites (configs/<name>.env). Used by the settings page.

Writing .env
    The new file is .env.example with the user's values filled in: an entry the user set becomes an active line
    (`KEY=value   # description   [default: x]`) in its own place; an entry left at its built-in default stays a
    commented line, exactly as shipped. Family settings (BUDGET_SERVER_VRAM_RESERVE_GB, COLOUR_GEN, ...) go right
    after their family's line; anything .env.example doesn't know is kept in a block at the end, never dropped.
    The previous file is kept as .env.bak. Values are quoted only when needed, so python-dotenv and the built-in
    reader read every value back identically (a selftest checks that).

Favourites
    configs/<name>.env holds only the settings that differ from the shipped .env.example, each with the shipped
    value at save time (`# shipped: x`), plus a header with the tool version. Never secrets. Loading one onto a newer
    release: the new .env.example values, then the favourite on top; renamed settings follow envspec.LEGACY,
    settings that no longer exist are reported, new settings keep their new defaults, and where the shipped value
    changed AND the favourite changed it too, both are reported so the user can choose.
"""
from __future__ import annotations

import re
import os
import shutil
from datetime import datetime
from pathlib import Path

from . import envspec

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")
_PLAIN = re.compile(r"^[^\s#'\"\\]+$")


class EnvValueError(ValueError):
    pass


def quote(v: str) -> str:
    """A .env value both python-dotenv and envspec.read_env_file read back as `v`."""
    v = "" if v is None else str(v)
    if "\n" in v or "\r" in v:
        raise EnvValueError("a value cannot span several lines")
    if v and _PLAIN.match(v):
        return v
    if "'" not in v:
        return f"'{v}'"                      # single quotes: literal for python-dotenv and the built-in reader
    if '"' not in v and "\\" not in v:
        return f'"{v}"'
    raise EnvValueError("a value cannot contain both kinds of quote (or a ' and a backslash)")


def shipped_values(entries=None) -> dict:
    """{KEY: value .env.example ships} for every documented setting: the active line's value, or the built-in
    default of a commented line ("" = not set)."""
    entries = entries if entries is not None else envspec.parse_example()
    return {e.key: (e.value if e.active else e.default) for e in entries if not e.family}


def _line(key, value, comment):
    head = f"{key}={quote(value)}"
    return (head.ljust(38) + " # " + comment) if comment else head


def render_env(values: dict, example_path=envspec.EXAMPLE_PATH, stamp: str | None = None) -> str:
    """The text of a .env holding `values` ({KEY: str}) in the .env.example layout."""
    text = Path(example_path).read_text(encoding="utf-8")
    lines = text.splitlines()
    entries = envspec.parse_example(example_path)
    documented = {e.key: e for e in entries if not e.family}
    families = [e for e in entries if e.family]
    values = {envspec.LEGACY.get(k, k): ("" if v is None else str(v)) for k, v in values.items()}
    code = envspec.code_settings()
    out = list(lines)
    after: dict[int, list[str]] = {}
    extra = []
    for k, v in values.items():
        e = documented.get(k)
        if e is not None:
            if not e.active and envspec.same_default(v, *code.get(k, (e.default, "str"))):
                continue                                    # at the built-in default: the commented line stays
            raw = lines[e.line - 1]
            _, comment = envspec.split_value(raw.split("=", 1)[1], dotenv=not raw.lstrip().startswith("#"))
            if not e.active or "[default:" not in comment:
                comment = (e.desc + f"   [default: {e.default or 'empty'}]").strip()
            out[e.line - 1] = _line(k, v, comment)
            continue
        fam = next((f for f in families if f.pattern().match(k)), None)
        if fam is not None:
            after.setdefault(fam.line, []).append(_line(k, v, ""))
        else:
            extra.append(_line(k, v, ""))
    final = []
    for i, line in enumerate(out, start=1):
        final.append(line)
        final.extend(after.get(i, []))
    head = [f"# Written by the settings page (python benchmark.py --ui) on {stamp or datetime.now():%Y-%m-%d %H:%M}; "
            "the previous file is .env.bak"]
    tail = (["", "# --- Not in .env.example (kept from your .env; --check-env says why each is unknown) ---"] + extra
            if extra else [])
    return "\n".join(head + final + tail) + "\n"


def read_values(path) -> dict:
    """{KEY: value} of a .env (the last line wins, like python-dotenv). {} when it does not exist."""
    p = Path(path)
    if not p.is_file():
        return {}
    out = {}
    for k, v, _ in envspec.read_env_file(p, blank_comments=True):   # 'KEY=   # text' is empty, not '# text'
        out[k] = v
    return out


def write_env(values: dict, env_path, example_path=envspec.EXAMPLE_PATH) -> Path | None:
    """Write `values` to env_path in the example layout; the old file goes to <env_path>.bak. Returns the backup."""
    env_path = Path(env_path)
    text = render_env(values, example_path)
    bak = None
    if env_path.exists():
        bak = env_path.with_name(env_path.name + ".bak")
        shutil.copy2(env_path, bak)
        private(bak)
    tmp = env_path.with_name(env_path.name + ".tmp")
    # v2.3.1: .env holds tokens. Created readable by its owner only (0600), whatever the umask is: on a server share
    # the default was readable (often writable) by everyone who can reach the share.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    private(tmp)
    tmp.replace(env_path)
    return bak


def private(path):
    """chmod 0600, where the file system has such a thing (not fatal on a share or on Windows)."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def defaults_with_secrets(current: dict, example_path=envspec.EXAMPLE_PATH) -> dict:
    """'Return to defaults': the shipped .env.example's active values, keeping the secrets (tokens, keys) set now."""
    entries = envspec.parse_example(example_path)
    vals = {e.key: e.value for e in entries if e.active and not e.family}
    vals.update({k: v for k, v in current.items() if k in envspec.SECRETS and v})
    return vals


# --------------------------------------------------------------------------- favourites
def favourite_path(configs_dir, name) -> Path:
    if not NAME_RE.match(name or ""):
        raise EnvValueError("a favourite name is 1-40 letters, digits, '.', '_' or '-'")
    return Path(configs_dir) / f"{name}.env"


def list_favourites(configs_dir) -> list[str]:
    d = Path(configs_dir)
    return sorted(p.stem for p in d.glob("*.env") if NAME_RE.match(p.stem)) if d.is_dir() else []


def favourite_text(name, values: dict, version: str, example_path=envspec.EXAMPLE_PATH) -> str:
    shipped = shipped_values(envspec.parse_example(example_path))
    lines = [f"# Local LLM Benchmark Suite favourite: {name}",
             f"# Saved with v{version} on {datetime.now():%Y-%m-%d %H:%M}. Only the settings that differ from the "
             "shipped .env.example; never secrets.",
             "# Load it on the settings page (python benchmark.py --ui). '# shipped:' = the .env.example value when "
             "it was saved."]
    for k in sorted(values):
        v = "" if values[k] is None else str(values[k])
        if k in envspec.SECRETS or v == shipped.get(k, ""):
            continue
        lines.append(f"{k}={quote(v)}".ljust(38) + f" # shipped: {shipped.get(k, '')}")
    return "\n".join(lines) + "\n"


def save_favourite(configs_dir, name, values: dict, version: str, example_path=envspec.EXAMPLE_PATH) -> Path:
    p = favourite_path(configs_dir, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(favourite_text(name, values, version, example_path), encoding="utf-8")
    return p


def read_favourite(path) -> tuple[dict, dict]:
    """({KEY: value}, {KEY: shipped value when saved}) from a favourite file."""
    vals, then = {}, {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z][A-Z0-9_]*)=(.*)$", raw.strip())
        if not m:
            continue
        v, comment = envspec.split_value(m.group(2))
        vals[m.group(1)] = v
        if comment.startswith("shipped:"):
            then[m.group(1)] = comment[len("shipped:"):].strip()
    return vals, then


def apply_favourite(path, current: dict, example_path=envspec.EXAMPLE_PATH) -> dict:
    """Merge a favourite onto today's .env.example: {"values", "renamed", "removed", "conflicts"}.
    Secrets come from `current` (never from the favourite). A conflict keeps the favourite's value until the user
    chooses."""
    entries = envspec.parse_example(example_path)
    shipped = shipped_values(entries)
    families = [e for e in entries if e.family]
    code = envspec.code_settings()
    fav, then = read_favourite(path)
    values = {e.key: e.value for e in entries if e.active and not e.family}
    values.update({k: v for k, v in current.items() if k in envspec.SECRETS and v})
    renamed, removed, conflicts = [], [], []
    for k, v in fav.items():
        if k in envspec.SECRETS:
            continue
        if k in envspec.LEGACY:
            renamed.append((k, envspec.LEGACY[k]))
            then[envspec.LEGACY[k]] = then.get(k, "")
            k = envspec.LEGACY[k]
        if k not in shipped and k not in code and not any(f.pattern().match(k) for f in families):
            removed.append((k, v))
            continue
        now = shipped.get(k, "")
        if k in then and then[k] != now and v != now:
            conflicts.append({"key": k, "favourite": v, "shipped_then": then[k], "shipped_now": now})
        values[k] = v
    return {"values": values, "renamed": renamed, "removed": removed, "conflicts": conflicts}
