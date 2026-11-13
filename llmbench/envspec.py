"""The settings file: parse .env.example, find the settings the code reads, and check a user's .env against both.

.env.example is the one documented list of settings. Its format (described at the top of the file) is strict enough
to be read by a program, so `--check-env` (and later the settings page) can show every setting with its part,
section, description and built-in default:

    KEY=value      # description   [default: x]     active line: the value is used; [default] = the built-in value
    # KEY=value    # description                    commented line: the value shown IS the built-in default
    # KEY=         # description                    commented and empty: not set by default
    # KEY_<NAME>=  # description                    a family of settings (BUDGET_<PROFILE>_<RULE>, ...)

A comment line right after an entry whose text starts with 10+ spaces continues that entry's description.

The code side is found by reading the package's own source (ast): every _env / _env_list / _env_ints / _env_tokens /
_env_budget / os.getenv call with a literal name, plus the per-dataset switches from EVAL_SPECS. A selftest checks
that both lists agree, so a new setting cannot be added without documenting it, and a documented default cannot drift
from the code.

Nothing here changes how settings are loaded; it only reads files.
"""
from __future__ import annotations

import ast
import difflib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = PKG_DIR.parent
EXAMPLE_PATH = SCRIPT_DIR / ".env.example"

TRUE_WORDS = ("1", "true", "yes", "on")
FALSE_WORDS = ("0", "false", "no", "off")

# Read by the code but not user settings (set by the tool for its own subprocesses, or by the OS).
INTERNAL = {"LLMB_MEM_MB", "PATH", "SYSTEMROOT", "DISPLAY", "WAYLAND_DISPLAY"}
# Old names that still work, -> the current name.
LEGACY = {"STRATA_SWAP_URL": "SECONDARY_URL",                       # v2.3 name -> current
          # v2.3.5: the endpoint settings are named for their role, not for the products first tested
          "LLAMA_SWAP_URL": "SERVER_URL", "LLAMA_SWAP_API_KEY": "SERVER_API_KEY",
          "STRATA_URL": "SECONDARY_URL", "STRATA_API_KEY": "SECONDARY_API_KEY", "STRATA_MODEL": "SECONDARY_MODEL",
          "STRATA_PARAMS_B": "SECONDARY_PARAMS_B", "STRATA_ACTIVE_B": "SECONDARY_ACTIVE_B",
          "STRATA_QUANT": "SECONDARY_QUANT", "STRATA_CTX": "SECONDARY_CTX"}


def old_names(name):
    """The old spellings of a current setting name (LEGACY reversed), e.g. SERVER_URL -> ["LLAMA_SWAP_URL"]."""
    return [old for old, new in LEGACY.items() if new == name]
# Defaults the source scan can't see (written as `os.getenv(X) or 900`).
EXTRA_DEFAULTS = {"ESTIMATE_GPU_BW_GBS": ("auto", "str"),
                  # v2.3 settings page (ui/server.py reads them from .env + the shell itself, never via load_config)
                  "UI_HOST": (None, "str"), "UI_PORT": (41010, "int"), "UI_TOKEN": (None, "str")}
# v2.3: settings with a fixed set of values (load_config() stops on anything else; a selftest checks both agree).
# Matched case-insensitively, like load_config.
_KV = ("f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "iq4_nl", "q4_0")
CHOICES = {
    "EVAL_PROFILE": ("test", "quick", "basic", "moderate", "standard", "thorough", "deep", "ultra", "full", "funnel"),
    "EVAL_FUNNEL_FINAL": ("basic", "moderate", "standard", "thorough", "deep", "ultra", "full"),
    "EVAL_SKIP_INELIGIBLE": ("auto", "true", "false"),
    "EVAL_RESULT_CACHE": ("true", "refresh", "false"),
    "EVAL_SQ_BASIS": ("auto", "measured", "single"),
    "EVAL_HARD_FAIL": ("stop", "continue"),
    "GPU_BACKEND": ("auto", "nvml", "smi"),
    "EVAL_HTTP_BACKEND": ("auto", "aiohttp", "threads"),
    "EVAL_CODE_SANDBOX": ("auto", "docker", "subprocess"),
    "MIN_CTX_ACTION": ("flag", "refuse"),
    "LONGCTX_KV_ACTION": ("flag", "skip"),
    "LONGCTX_KV_MIN": _KV,
    "GOLDILOCKS_MODE": ("linear", "log"),
    "COLOUR_STYLE": ("bands", "smooth"),
    "COLOUR_MODE": ("absolute", "relative"),
}
# v2.3: tokens and keys. The settings page masks them; they never go into favourites, logs or dashboards.
SECRETS = ("SERVER_API_KEY", "HF_TOKEN", "SECONDARY_API_KEY", "HA_TOKEN", "UI_TOKEN")

_ENTRY = re.compile(r"^(?P<hash># ?)?(?P<key>[A-Z][A-Z0-9_]*(?:<[A-Z]+>[A-Z0-9_]*)*)=(?P<rest>.*)$")
_DEFAULT_TAG = re.compile(r"\[default:\s*(?P<d>[^\]]*)\]\s*$")
_PART = re.compile(r"^#\s+PART\s+(?P<n>\d)\s+-\s+(?P<title>.+?)\s*$")
_SECTION = re.compile(r"^#\s*---\s*(?P<title>.+?)\s*---\s*$")


# --------------------------------------------------------------------------- #
# Reading .env-style lines (same rules as python-dotenv for the simple lines this tool uses)
# --------------------------------------------------------------------------- #

def split_value(rest: str, dotenv: bool = True) -> tuple[str, str]:
    """'value   # comment' -> ('value', 'comment'). A quoted value keeps what is inside the quotes.

    dotenv=True follows python-dotenv exactly: a comment needs whitespace before '#' and something before that, so
    'KEY=   # comment' gives the value '# comment' (write KEY="" for an empty value with a comment). dotenv=False is
    for the commented-out lines of .env.example, where '# KEY=   # text' means "empty, described by text"."""
    rest = rest.strip()
    if rest and rest[0] in "\"'":
        end = rest.find(rest[0], 1)
        if end > 0:
            tail = rest[end + 1:].strip()
            return rest[1:end], tail[1:].strip() if tail.startswith("#") else ""
        return rest[1:], ""
    if rest.startswith("#") and not dotenv:        # '# KEY=   # text' in .env.example: empty value
        return "", rest[1:].strip()
    parts = re.split(r"\s+#", rest, maxsplit=1)
    return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")


def read_env_file(path, blank_comments=False) -> list[tuple[str, str, int]]:
    """Active KEY=value lines of a .env file, in order, as (key, value, line number). Duplicates are kept.

    python-dotenv reads 'KEY=   # comment' as the VALUE '# comment' (a real case: UI_TOKEN). With blank_comments=True,
    a value that is only a comment (whitespace, then '#', after the '=') is read as empty, which is what the line
    means. 'KEY=#abc' (no space) and quoted values are untouched."""
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if s.startswith("export "):
            s = s[7:].lstrip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            continue
        val = split_value(v)[0]
        if blank_comments and re.match(r"\s+#", v):
            val = ""
        out.append((k, val, i))
    return out


# --------------------------------------------------------------------------- #
# .env.example
# --------------------------------------------------------------------------- #

@dataclass
class Entry:
    key: str                     # KEY, or a family pattern like BUDGET_<PROFILE>_<RULE>
    value: str                   # the value written on the line
    active: bool                 # uncommented line
    default: str                 # built-in default as text ("" = not set)
    desc: str
    part: int                    # 1 setup, 2 run, 3 optional
    part_title: str
    section: str
    line: int
    family: bool = False         # KEY contains <...>

    @property
    def tier(self):
        return {1: "setup", 2: "run", 3: "optional"}.get(self.part, "optional")

    def pattern(self):
        """Regex for the real keys of a family entry."""
        return re.compile("^" + re.sub(r"<[A-Z]+>", "[A-Z0-9_]+?", self.key) + "$")


def parse_example(path=EXAMPLE_PATH) -> list[Entry]:
    entries: list[Entry] = []
    part, part_title, section = 0, "", ""
    last = None
    for i, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.rstrip()
        m = _PART.match(line)
        if m:
            part, part_title, section, last = int(m["n"]), m["title"], "", None
            continue
        m = _SECTION.match(line)
        if m:
            section, last = m["title"], None
            continue
        cont = re.match(r"^#( {10,})(\S.*)$", line)
        if last is not None and cont:
            last.desc = (last.desc + " " + cont.group(2).strip()).strip()
            continue
        m = _ENTRY.match(line) if part else None          # the explanation above PART 1 holds no settings
        if m:
            value, comment = split_value(m["rest"], dotenv=not m["hash"])
            tag = _DEFAULT_TAG.search(comment)
            if tag:
                comment = comment[:tag.start()].rstrip()
                default = tag["d"].strip()
                default = "" if default.lower() == "empty" else default
            else:
                default = value
            last = Entry(key=m["key"], value=value, active=not m["hash"], default=default, desc=comment,
                         part=part, part_title=part_title, section=section, line=i, family="<" in m["key"])
            entries.append(last)
            continue
        last = None
    return entries


# --------------------------------------------------------------------------- #
# What the code reads
# --------------------------------------------------------------------------- #

_READERS = {"_env": None, "_env_list": "list", "_env_ints": "ints", "_env_tokens": "tokens", "_env_budget": "budget"}


def _cast_name(node):
    if isinstance(node, ast.Name):
        return node.id                         # bool / int / float / str
    return "str"


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def code_settings() -> dict[str, tuple[object, str]]:
    """{NAME: (default, kind)} for every setting the package reads with a literal name. kind: bool, int, float, str,
    list, ints, tokens, budget."""
    found: dict[str, tuple[object, str]] = {}
    for py in sorted(PKG_DIR.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            first = node.args[0]
            if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                continue
            name = first.value
            if not re.fullmatch(r"[A-Z][A-Z0-9_]+", name):
                continue
            f = node.func
            fname = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if fname in _READERS:
                default = _literal(node.args[1]) if len(node.args) > 1 else None
                kind = _READERS[fname] or (_cast_name(node.args[2]) if len(node.args) > 2 else "str")
                if fname == "_env_ints" and default is None:
                    default = ()
            elif fname in ("getenv", "get") and isinstance(f, ast.Attribute) and (
                    (isinstance(f.value, ast.Name) and f.value.id == "os")
                    or (isinstance(f.value, ast.Attribute) and f.value.attr == "environ")):
                default, kind = (_literal(node.args[1]) if len(node.args) > 1 else None), "str"
            else:
                continue
            if name not in found or found[name][0] is None:
                found[name] = (default, kind)
    for name, dk in EXTRA_DEFAULTS.items():
        found[name] = dk
    from .config import EVAL_SPECS                      # per-dataset switches are read from EVAL_SPECS
    for spec in EVAL_SPECS.values():
        found.setdefault(spec["env"], (False, "bool"))
        if not spec.get("builtin"):
            found.setdefault(spec["env"] + "_REVISION", (None, "str"))
    for name in INTERNAL | set(LEGACY):
        found.pop(name, None)
    for name in CHOICES:                                 # v2.3 kinds for the settings page and --check-env
        if name in found:
            found[name] = (found[name][0], "choice")
    for name in SECRETS:
        if name in found:
            found[name] = (found[name][0], "secret")
    return found


def budget_defaults():
    """{BUDGET_<PROFILE>_<RULE>: default} for the built-in profiles, and the list of rule names."""
    from .config import BUDGET_DEFAULTS, BUDGET_RULES
    out = {}
    for prof, rules in BUDGET_DEFAULTS.items():
        for rule, val in rules.items():
            out[f"BUDGET_{prof.upper()}_{rule}"] = val
    return out, BUDGET_RULES


def same_default(doc: str, code, kind: str) -> bool:
    """Is the default written in .env.example the same as the code's?"""
    doc = (doc or "").strip()
    if code is None or code == "" or code == ():
        return doc == "" or (kind == "list" and doc.lower() in ("all", "*"))
    if kind == "bool":
        return doc.lower() in (TRUE_WORDS if code else FALSE_WORDS)
    if kind == "budget" and code == "auto":
        return doc.lower() in ("auto", "dynamic")
    if kind in ("int", "float", "budget") or isinstance(code, (int, float)) and not isinstance(code, bool):
        try:
            return float(doc) == float(code)
        except ValueError:
            return False
    if kind == "ints":
        try:
            vals = tuple(int(float(x.strip().lower().rstrip("k")) * (1024 if x.strip().lower().endswith("k") else 1))
                         for x in doc.split(",") if x.strip())
        except ValueError:
            return False
        return vals == tuple(code)
    return doc == str(code)


# --------------------------------------------------------------------------- #
# Checking a user's .env
# --------------------------------------------------------------------------- #

@dataclass
class EnvReport:
    env_path: Path
    example_path: Path
    found: bool = True
    unknown: list = field(default_factory=list)       # (key, value, line, suggestion)
    legacy: list = field(default_factory=list)        # (key, new_key, line)
    invalid: list = field(default_factory=list)       # (key, value, line, problem)
    duplicates: list = field(default_factory=list)    # (key, [lines], value used)
    shell: list = field(default_factory=list)         # (key, shell value, .env value)
    missing: list = field(default_factory=list)       # Entry (part 1/2 settings the .env doesn't have)
    missing_optional: int = 0
    changed: list = field(default_factory=list)       # (key, value, default, Entry)
    at_default: list = field(default_factory=list)    # (key, value)
    info: list = field(default_factory=list)          # free-text notes

    @property
    def problems(self):
        return len(self.unknown) + len(self.invalid)


def _value_problem(value, kind, key=None):
    v = (value or "").strip()
    if v == "":
        return None
    if kind == "choice" and key in CHOICES and v.lower() not in CHOICES[key]:
        return f"not one of {', '.join(CHOICES[key])}: the benchmark stops with an error"
    if key == "AGENT_CTX_CURVE":
        from .agent import parse_ctx_curve
        try:
            parse_ctx_curve(v)
        except ValueError as e:
            return f"{e}: the benchmark stops with an error"
    if key == "ESTIMATE_GPU_BW_GBS":
        from .gpubw import valid
        why = valid(v)
        if why:
            return why
    if key == "UI_HOST":
        from .ui.server import clean_host
        h = clean_host(v)
        if h != v:
            return f"an address, not a URL: it is read as {h or '127.0.0.1'} (0.0.0.0 = open on the LAN)"
    if key == "DASH_FAMILIES":
        from .reporting.colours import parse_families
        try:
            parse_families(v)
        except ValueError as e:
            return f"{e}: the benchmark stops with an error"
    if kind == "bool" and v.lower() not in TRUE_WORDS + FALSE_WORDS:
        return "not true/false: it is read as false"
    if kind == "int":
        try:
            int(v)
        except ValueError:
            return "not a whole number: the benchmark stops with an error"
    if kind == "float":
        try:
            float(v)
        except ValueError:
            return "not a number: the benchmark stops with an error"
    if kind == "tokens" and v.lower() not in ("auto", "dynamic"):
        try:
            int(v)
        except ValueError:
            return "not a number of tokens (or empty for dynamic): the benchmark stops with an error"
    if kind == "budget" and v.lower() not in ("off", "none", "unlimited", "false", "no", "auto", "dynamic"):
        try:
            int(v)
        except ValueError:
            return "not a number of tokens (0 = unlimited, auto = a share of the context): the benchmark stops with an error"
    if kind == "ints":
        for x in v.split(","):
            x = x.strip().lower()
            if x and x not in ("0", "off", "none", "false"):
                try:
                    float(x.rstrip("k"))
                except ValueError:
                    return f"{x!r} is not a token count (8192 or 8k): the benchmark stops with an error"
    return None


def check_env(env_path=None, example_path=EXAMPLE_PATH, environ=None) -> EnvReport:
    env_path = Path(env_path) if env_path else SCRIPT_DIR / ".env"
    environ = os.environ if environ is None else environ
    rep = EnvReport(env_path=env_path, example_path=Path(example_path))
    entries = parse_example(example_path)
    documented = {e.key: e for e in entries if not e.family}
    families = [e for e in entries if e.family]
    code = code_settings()
    bdefaults, brules = budget_defaults()
    if not env_path.exists():
        rep.found = False
        return rep
    lines = read_env_file(env_path)
    seen: dict[str, list] = {}
    for k, v, n in lines:
        seen.setdefault(k, []).append((v, n))
    values = {k: occ[-1][0] for k, occ in seen.items()}              # python-dotenv: the last line wins
    for k, occ in seen.items():
        if len(occ) > 1:
            rep.duplicates.append((k, [n for _, n in occ], occ[-1][0]))
    known = set(documented) | set(code)
    profile = (values.get("BUDGET_PROFILE") or environ.get("BUDGET_PROFILE") or "none").strip().lower()
    for k, occ in seen.items():
        v, n = occ[-1]
        if k in environ and environ[k] != v:
            rep.shell.append((k, environ[k], v))
        if k in LEGACY:
            rep.legacy.append((k, LEGACY[k], n))
            continue
        if v.startswith("#"):
            rep.invalid.append((k, v, n, 'only a comment after "=": this tool reads it as empty, but python-dotenv in '
                                         f'another program reads it as text. Write {k}="" or put the comment on its own line'))
            continue
        if k in known:
            kind = code.get(k, (None, "str"))[1]
            prob = _value_problem(v, kind, k)
            if prob:
                rep.invalid.append((k, v, n, prob))
            continue
        fam = next((e for e in families if e.pattern().match(k)), None)
        if fam is not None and k.startswith("BUDGET_"):
            rule = next((r for r in sorted(brules, key=len, reverse=True) if k.endswith("_" + r)), None)
            if rule is None:
                sug = difflib.get_close_matches(k.rsplit("_", 1)[-1], brules, n=1)
                rep.unknown.append((k, v, n, f"budget rule; known rules: {', '.join(brules)}"
                                    if not sug else f"budget rule {sug[0]}?"))
                continue
            prof = k[len("BUDGET_"):-len(rule) - 1].lower()
            if prof != profile:
                rep.info.append(f"{k} applies only with BUDGET_PROFILE={prof} (now: {profile})")
            continue
        if fam is not None and k.startswith("EVAL_THINK_BUDGET_"):
            from .config import EVAL_SPECS
            names = [s["env"][len("EVAL_"):] for s in EVAL_SPECS.values()]
            if k[len("EVAL_THINK_BUDGET_"):] not in names:
                sug = difflib.get_close_matches(k[len("EVAL_THINK_BUDGET_"):], names, n=1)
                rep.unknown.append((k, v, n, f"EVAL_THINK_BUDGET_{sug[0]}?" if sug else
                                    "dataset names: " + ", ".join(names)))
                continue
            prob = _value_problem(v, "budget")
            if prob:
                rep.invalid.append((k, v, n, prob))
            continue
        if fam is not None and k.startswith("COLOUR_"):
            from .reporting.colours import SCALES, parse_thresholds
            metric = k[len("COLOUR_"):]
            if metric not in SCALES:
                sug = difflib.get_close_matches(metric, list(SCALES), n=1)
                rep.unknown.append((k, v, n, f"COLOUR_{sug[0]}?" if sug else "metrics: " + ", ".join(SCALES)))
                continue
            try:
                parse_thresholds(metric, v)
            except ValueError as e:
                rep.invalid.append((k, v, n, str(e).split(": ", 1)[-1]))
            continue
        if fam is not None:
            continue
        sug = difflib.get_close_matches(k, sorted(known), n=1, cutoff=0.75)
        rep.unknown.append((k, v, n, f"{sug[0]}?" if sug else ""))
    for e in entries:
        if e.family:
            continue
        if e.key not in values:
            if e.part in (1, 2) and e.active:
                rep.missing.append(e)
            else:
                rep.missing_optional += 1
            continue
        v = values[e.key]
        code_default, kind = code.get(e.key, (None, "str"))
        if same_default(v, code_default, kind):
            rep.at_default.append((e.key, v))
        else:
            rep.changed.append((e.key, v, e.default, e))
    for k in values:                                   # budget overrides: compare with the built-in profile value
        if k in bdefaults:
            rep.info.append(f"{k}={values[k]} (built-in {bdefaults[k]:g})" if isinstance(bdefaults[k], float)
                            else f"{k}={values[k]} (built-in {bdefaults[k]})")
    return rep


def _shown(v):
    return v if v != "" else "(empty)"


def print_report(rep: EnvReport) -> int:
    """Print the check; returns the exit code (1 if a setting is unknown or has an unusable value)."""
    print(f".env check: {rep.env_path}")
    print(f"  against {rep.example_path.name} (the documented settings of this version)")
    if not rep.found:
        print(f"\n  No .env found. Create one:  cp {rep.example_path.name} .env   and edit PART 1 and PART 2.")
        return 1

    def block(title, rows):
        if rows:
            print(f"\n  {title}")
            for r in rows:
                print("    " + r)

    block("Unknown settings: the benchmark ignores these (typo, or removed in this version):",
          [f"line {n}: {k}={v}" + (f"   -> did you mean {s}" if s.endswith("?") else f"   ({s})" if s else "")
           for k, v, n, s in rep.unknown])
    block("Values that will not work as intended:",
          [f"line {n}: {k}={v}   -> {p}" for k, v, n, p in rep.invalid])
    block("Old names (still read; rename when convenient):",
          [f"line {n}: {k} -> {new}" for k, new, n in rep.legacy])
    block("Set more than once (the last line wins):",
          [f"{k}: lines {', '.join(map(str, ns))}; used: {_shown(v)}" for k, ns, v in rep.duplicates])
    block("Set in your shell environment too (the shell value wins over .env):",
          [f"{k}: shell {_shown(s)}, .env {_shown(v)}" for k, s, v in rep.shell])
    block("Not in your .env (PART 1 / PART 2 settings; the built-in default is used):",
          [f"{e.key}   default {_shown(e.default)}"
           + (f"; {rep.example_path.name} suggests {_shown(e.value)}" if e.value != e.default else "")
           + f"   # {e.desc}" for e in rep.missing])
    block("Changed from the built-in default:",
          [f"{k}={_shown(v)}   (default {_shown(d)})" for k, v, d, _ in rep.changed])
    block("Notes:", rep.info)
    print(f"\n  {len(rep.at_default)} setting(s) in your .env equal the built-in default; "
          f"{rep.missing_optional} optional setting(s) are not in it and use their defaults.")
    if rep.problems:
        print(f"\n  {rep.problems} problem(s) found.")
        return 1
    print("\n  No problems found.")
    return 0


def check_env_cli(path=None) -> int:
    return print_report(check_env(path))
