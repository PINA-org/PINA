#!/usr/bin/env python3
"""
check_skill_sync.py

Scans skill SKILL.md files for references to functions/classes/modules
(written as backtick-wrapped identifiers or inside fenced code blocks) and
checks whether those identifiers still exist in the actual codebase.

Usage:
    # Quick overview across every skill (counts only, fast)
    python check_skill_sync.py --skills-dir skills --code-dir src --overview

    # Deep check of ONE skill, with fuzzy suggestions for anything stale
    python check_skill_sync.py --skills-dir skills --code-dir src --skill my-skill-name

    # Dump full JSON (for the calling Claude session to parse/reason over)
    python check_skill_sync.py --skills-dir skills --code-dir src --skill my-skill-name --json-out report.json

Notes / heuristics (read this before trusting the output blindly):
- We only flag tokens that "look like code": they contain an underscore, mix
  upper/lower case, are dotted (module.func), or end in "()". Plain lowercase
  English words in backticks (e.g. `output`, `example`) are ignored on
  purpose to keep noise down. This means genuinely single-word, all-lowercase
  identifiers (e.g. a function literally named `parse`) can be missed.
- "not_found" means the token wasn't found anywhere in --code-dir, not that
  it's definitely wrong -- it may live in a directory you didn't point at,
  or be a genuinely external/library symbol. Always sanity check before
  editing a skill.
- This is intentionally dependency-free (no ripgrep/ctags required) so it
  runs anywhere Python 3 does. It will be slower on very large repos.
"""

import argparse
import difflib
import json
import os
import re
import sys
from collections import defaultdict

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rb", ".java", ".kt",
    ".rs", ".cpp", ".cc", ".c", ".h", ".hpp", ".cs", ".php", ".swift",
    ".scala", ".m", ".mm",
}

EXCLUDE_DIRS = {
    ".git", "node_modules", "venv", ".venv", "dist", "build", "__pycache__",
    ".mypy_cache", ".pytest_cache", "target", "vendor", "out",
}

# Patterns that pull out a *defining* occurrence of a symbol (best-effort,
# multi-language, intentionally permissive rather than a real parser).
DEFINITION_PATTERNS = [
    re.compile(r"^\s*(?:async\s+)?def\s+(\w+)", re.M),                 # python func
    re.compile(r"^\s*class\s+(\w+)", re.M),                            # python/ruby/js class
    re.compile(r"\bfunction\s+(\w+)\s*\("),                            # js function
    re.compile(r"\bconst\s+(\w+)\s*="),                                # js/ts const
    re.compile(r"\blet\s+(\w+)\s*="),                                  # js/ts let
    re.compile(r"\bexport\s+(?:default\s+)?(?:function|class)\s+(\w+)"),
    re.compile(r"\bfunc\s+(?:\([^)]*\)\s*)?(\w+)\s*\("),               # go func
    re.compile(r"\btype\s+(\w+)\s+struct\b"),                         # go struct
    re.compile(r"\bfn\s+(\w+)\s*\("),                                  # rust fn
    re.compile(r"\bstruct\s+(\w+)\b"),                                 # rust/c struct
    re.compile(r"\benum\s+(\w+)\b"),                                   # rust/java enum
    re.compile(r"\btrait\s+(\w+)\b"),                                  # rust trait
    re.compile(r"\bmodule\s+(\w+)\b"),                                 # ruby module
    re.compile(r"\binterface\s+(\w+)\b"),                              # ts/java interface
    re.compile(r"^[ \t]*(?:public|private|protected|static|[ \t])*class\s+(\w+)", re.M),  # java/c#
]

STOPWORDS = {
    "true", "false", "none", "null", "this", "that", "note", "example",
    "output", "input", "todo", "step", "file", "path", "code", "value",
    "result", "data", "config", "error", "warning", "info", "debug",
    "return", "returns", "params", "param", "args", "kwargs", "self",
}

INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
FENCED_BLOCK_RE = re.compile(r"```.*?```", re.S)
IDENT_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*(\(\))?$")


def looks_like_code(token):
    """Heuristic: does this backtick-wrapped token look like a real
    code identifier rather than an incidental English word?"""
    bare = token.rstrip("()")
    if not IDENT_TOKEN_RE.match(token):
        return False
    if bare.lower() in STOPWORDS:
        return False
    if "_" in bare:
        return True
    if "." in bare:
        return True
    if token.endswith("()"):
        return True
    if bare != bare.lower() and bare != bare.upper():
        return True  # mixed case -> camelCase / PascalCase
    if bare.isupper() and len(bare) > 2:
        return True  # CONSTANT_LIKE
    return False


def extract_candidates(skill_md_text):
    """Pull code-like identifiers out of a SKILL.md's prose and code blocks."""
    candidates = set()

    # Inline backtick spans (outside fenced blocks, to avoid double counting
    # weirdly, but overlap here is harmless -- we just dedupe into a set)
    stripped = FENCED_BLOCK_RE.sub("", skill_md_text)
    for m in INLINE_CODE_RE.finditer(stripped):
        token = m.group(1).strip()
        if looks_like_code(token):
            candidates.add(token)

    # Fenced code blocks: pull plausible identifiers via def/class/func-call
    # style patterns, reusing the same "looks like code" bar for tokens we
    # can't classify as a definition.
    for block in FENCED_BLOCK_RE.findall(skill_md_text):
        for pat in DEFINITION_PATTERNS:
            for m in pat.finditer(block):
                candidates.add(m.group(1))
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", block):
            tok = m.group(1)
            if looks_like_code(tok + "()"):
                candidates.add(tok)

    return sorted(candidates)


def iter_code_files(code_dir):
    for root, dirs, files in os.walk(code_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for f in files:
            if os.path.splitext(f)[1] in CODE_EXTENSIONS:
                yield os.path.join(root, f)


def build_index(code_dir):
    """Returns (definitions: symbol -> [(file, line), ...], all_tokens: set)"""
    definitions = defaultdict(list)
    all_tokens = set()
    token_re = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

    for path in iter_code_files(code_dir):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        text = "".join(lines)

        for pat in DEFINITION_PATTERNS:
            for m in pat.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                loc = (path, line_no)
                if loc not in definitions[m.group(1)]:
                    definitions[m.group(1)].append(loc)

        for m in token_re.finditer(text):
            all_tokens.add(m.group(0))

    return definitions, all_tokens


def resolve_candidate(token, definitions, all_tokens):
    bare = token.rstrip("()")
    last_part = bare.split(".")[-1]

    if last_part in definitions:
        return "defined", definitions[last_part]
    if last_part in all_tokens:
        return "used_only", []
    return "not_found", []


def suggest(token, definitions):
    bare = token.rstrip("()").split(".")[-1]
    matches = difflib.get_close_matches(bare, definitions.keys(), n=3, cutoff=0.6)
    return matches


def find_skill_dirs(skills_dir):
    result = []
    if not os.path.isdir(skills_dir):
        return result
    for name in sorted(os.listdir(skills_dir)):
        p = os.path.join(skills_dir, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "SKILL.md")):
            result.append((name, os.path.join(p, "SKILL.md")))
    return result


def check_one_skill(skill_name, skill_md_path, definitions, all_tokens):
    with open(skill_md_path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()

    candidates = extract_candidates(text)
    entries = []
    for token in candidates:
        status, locations = resolve_candidate(token, definitions, all_tokens)
        entry = {"token": token, "status": status}
        if status == "defined":
            unique_files = sorted({f for f, _ in locations})
            entry["locations"] = [f"{f}:{min(ln for f2, ln in locations if f2 == f)}" for f in unique_files[:3]]
        if status == "not_found":
            sugg = suggest(token, definitions)
            if sugg:
                entry["suggestions"] = sugg
        entries.append(entry)

    return {
        "skill": skill_name,
        "path": skill_md_path,
        "checked": len(entries),
        "not_found": [e for e in entries if e["status"] == "not_found"],
        "used_only": [e for e in entries if e["status"] == "used_only"],
        "defined": [e for e in entries if e["status"] == "defined"],
        "entries": entries,
    }


def print_overview(results):
    print(f"{'SKILL':<30} {'checked':>8} {'defined':>8} {'used_only':>10} {'NOT FOUND':>10}")
    for r in results:
        print(f"{r['skill']:<30} {r['checked']:>8} {len(r['defined']):>8} "
              f"{len(r['used_only']):>10} {len(r['not_found']):>10}")
    flagged = [r for r in results if r["not_found"]]
    if flagged:
        print(f"\n{len(flagged)} skill(s) have possibly-stale references. "
              f"Re-run with --skill <name> for details.")
    else:
        print("\nNo stale-looking references found across any skill. Nice.")


def print_detail(result):
    print(f"# Sync report: {result['skill']}")
    print(f"({result['checked']} code-like references checked)\n")

    if result["not_found"]:
        print("## Possibly stale (not found anywhere in --code-dir)")
        for e in result["not_found"]:
            line = f"  - `{e['token']}`"
            if e.get("suggestions"):
                line += f"  -- did you mean: {', '.join(e['suggestions'])}?"
            print(line)
        print()
    else:
        print("## Nothing flagged as stale.\n")

    if result["used_only"]:
        print("## Found as usage but not as a definition (lower confidence)")
        for e in result["used_only"]:
            print(f"  - `{e['token']}`")
        print()

    if result["defined"]:
        print("## Confirmed defined in codebase")
        for e in result["defined"]:
            locs = "; ".join(e.get("locations", []))
            print(f"  - `{e['token']}`  ({locs})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skills-dir", required=True, help="Directory containing one subfolder per skill")
    ap.add_argument("--code-dir", required=True, help="Root of the actual codebase to check against")
    ap.add_argument("--skill", help="Only deep-check this one skill (folder name under --skills-dir)")
    ap.add_argument("--overview", action="store_true", help="Print counts for every skill instead of full detail")
    ap.add_argument("--json-out", help="Also write the full result set as JSON to this path")
    ap.add_argument("--fail-on-stale", action="store_true",
                     help="Exit with status 1 if any checked skill has not_found references (for CI)")
    args = ap.parse_args()

    skill_dirs = find_skill_dirs(args.skills_dir)
    if not skill_dirs:
        print(f"No skills with a SKILL.md found under {args.skills_dir}", file=sys.stderr)
        sys.exit(1)

    if args.skill:
        skill_dirs = [(n, p) for n, p in skill_dirs if n == args.skill]
        if not skill_dirs:
            print(f"Skill '{args.skill}' not found under {args.skills_dir}", file=sys.stderr)
            sys.exit(1)

    definitions, all_tokens = build_index(args.code_dir)

    results = [check_one_skill(name, path, definitions, all_tokens) for name, path in skill_dirs]

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)

    if args.overview or len(results) > 1:
        print_overview(results)
    else:
        print_detail(results[0])

    if args.fail_on_stale and any(r["not_found"] for r in results):
        print("\nFailing: at least one skill references code that no longer "
              "exists. Run this locally with --skill <name> to see details, "
              "then update the affected skill (never auto-fix in CI).",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()