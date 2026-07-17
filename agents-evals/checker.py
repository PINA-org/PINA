#!/usr/bin/env python3
"""
agents-evals checker — assertion engine for PINA skill evaluation.

Usage:
    python checker.py prompts/<prompt-file>.yaml transcripts/<transcript>.md

Output: JSON with per-assertion pass/fail and extracted code diagnostics.
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_transcript(path):
    with open(path) as f:
        return f.read()


def extract_code_blocks(text):
    """Extract all ```python ... ``` blocks from a transcript."""
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    return [b.strip() for b in blocks]


def check_assertion(assertion, transcript, code_blocks):
    """Run a single assertion against transcript and code blocks. Returns (passed, detail)."""
    atype = assertion["type"]
    values = assertion.get("values", [])
    min_count = assertion.get("min_count", 1)
    expect = assertion.get("expect", True)
    kwargs = assertion.get("kwargs", {})

    if atype == "text_contains":
        found = 0
        for val in values:
            if val.lower() in transcript.lower():
                found += 1
        passed = found >= min_count
        detail = f"found {found}/{min_count} required values in text" if found < min_count else "all values found in text"

    elif atype == "code_regex":
        code_text = "\n".join(code_blocks) if code_blocks else ""
        found = 0
        for pattern in values:
            if re.search(pattern, code_text, re.IGNORECASE | re.DOTALL):
                found += 1
        passed = found >= min_count
        detail = f"found {found}/{min_count} matching patterns in code" if found < min_count else "all patterns matched in code"

    elif atype == "code_imports":
        passed = False
        imports_found = []
        for block in code_blocks:
            try:
                tree = ast.parse(block)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports_found.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        for alias in node.names:
                            imports_found.append(f"{module}.{alias.name}")
            except SyntaxError:
                continue
        combined = " ".join(imports_found).lower()
        found = sum(1 for v in values if v.lower() in combined)
        passed = found >= min_count
        detail = f"found {found}/{min_count} required imports" if found < min_count else f"required imports present ({', '.join(values)})"

    elif atype == "code_compiles":
        if not code_blocks:
            passed = False
            detail = "no code blocks found to compile"
        else:
            failures = []
            for i, block in enumerate(code_blocks):
                try:
                    compile(block, f"block_{i}", "exec")
                except SyntaxError as e:
                    failures.append(f"block {i}: {e}")
            passed = len(failures) == 0
            detail = f"{len(code_blocks)} blocks compiled" if passed else f"compile failures: {'; '.join(failures)}"

    elif atype == "code_runs":
        if not code_blocks:
            passed = False
            detail = "no code blocks found to execute"
        else:
            max_epochs = kwargs.get("max_epochs", 2)
            failures = []
            namespace = {"__builtins__": __builtins__}
            for i, block in enumerate(code_blocks):
                try:
                    # Patch common training parameters to keep runs fast
                    patched_block = re.sub(
                        r"max_epochs\s*=\s*\d+",
                        f"max_epochs={max_epochs}",
                        block,
                    )
                    patched_block = re.sub(
                        r"Trainer\([^)]*\)",
                        lambda m: re.sub(
                            r"max_epochs\s*=\s*\d+",
                            f"max_epochs={max_epochs}",
                            m.group(0),
                        ),
                        patched_block,
                    )
                    exec(patched_block, namespace)
                except Exception as e:
                    failures.append(f"block {i}: {type(e).__name__}: {e}")
            passed = len(failures) == 0
            detail = f"{len(code_blocks)} blocks executed" if passed else f"execution failures: {'; '.join(failures)}"

    elif atype == "not_present":
        found = []
        for val in values:
            if val.lower() in transcript.lower():
                found.append(val)
        passed = len(found) == 0
        detail = f"unexpected values found: {found}" if found else "none of the forbidden values present"

    else:
        passed = False
        detail = f"unknown assertion type: {atype}"

    if not expect:
        passed = not passed
        detail = f"(inverted) {detail}"

    return passed, detail


def compute_skill_metrics(transcript, skills_expected):
    """Compute precision, recall, and F1 for skill triggering."""
    if not skills_expected:
        # Near-miss: we expect NO skills to load
        skill_names = [
            "pina-workflow", "create-problem", "define-domains",
            "define-equations", "condition-setup", "select-model",
            "select-solver", "select-trainer", "create-skill",
            "skill-sync-checker",
        ]
        any_loaded = any(
            name.lower() in transcript.lower()
            for name in skill_names
        )
        # For near-miss: precision = 1 if none loaded, else 0
        # recall is undefined (no relevant items), so we report N/A
        return {
            "precision": 1.0 if not any_loaded else 0.0,
            "recall": None,
            "f1": 1.0 if not any_loaded else 0.0,
            "true_positives": 0,
            "false_positives": sum(1 for n in skill_names if n.lower() in transcript.lower()),
            "false_negatives": 0,
        }

    # Normal case: check which expected skills were loaded
    loaded_skills = set()
    for name in skills_expected:
        if name.lower() in transcript.lower():
            loaded_skills.add(name)

    also_loaded = set()
    all_skill_names = [
        "pina-workflow", "create-problem", "define-domains",
        "define-equations", "condition-setup", "select-model",
        "select-solver", "select-trainer", "create-skill",
        "skill-sync-checker",
    ]
    for name in all_skill_names:
        if name.lower() in transcript.lower() and name not in skills_expected:
            also_loaded.add(name)

    expected_set = set(skills_expected)
    tp = len(loaded_skills)
    fp = len(also_loaded)
    fn = len(expected_set - loaded_skills)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "loaded": sorted(loaded_skills),
        "unexpected": sorted(also_loaded),
        "missing": sorted(expected_set - loaded_skills),
    }


def parse_transcript_filename(path):
    """Parse <prompt-id>--<model>--<skills>--<run>.md"""
    stem = Path(path).stem
    parts = stem.split("--")
    if len(parts) >= 4:
        return {
            "prompt_id": parts[0],
            "model": parts[1],
            "skills_mode": parts[2],
            "run": int(parts[3]) if parts[3].isdigit() else parts[3],
        }
    return {"prompt_id": stem, "model": "unknown", "skills_mode": "unknown", "run": 1}


def run_check(prompt_data, transcript_text, transcript_path=None):
    """Run all assertions for a prompt against a transcript."""
    prompt_id = prompt_data.get("id", "unknown")
    assertions = prompt_data.get("assertions", [])
    skills_expected = prompt_data.get("skills_expected", [])

    code_blocks = extract_code_blocks(transcript_text)
    n_code_blocks = len(code_blocks)

    compiler_results = []
    for i, block in enumerate(code_blocks):
        try:
            compile(block, f"block_{i}", "exec")
            compiler_results.append(True)
        except SyntaxError:
            compiler_results.append(False)

    runner_results = []
    namespace = {"__builtins__": __builtins__}
    for i, block in enumerate(code_blocks):
        try:
            patched = re.sub(r"max_epochs\s*=\s*\d+", "max_epochs=2", block)
            patched = re.sub(
                r"\baccelerator\s*=\s*['\"](gpu|cuda)['\"]",
                'accelerator="cpu"',
                patched,
            )
            exec(patched, namespace)
            runner_results.append(True)
        except Exception:
            runner_results.append(False)

    skill_metrics = compute_skill_metrics(transcript_text, skills_expected)

    assertion_results = []
    for a in assertions:
        passed, detail = check_assertion(a, transcript_text, code_blocks)
        assertion_results.append({
            "id": a.get("id", "unnamed"),
            "type": a.get("type"),
            "passed": passed,
            "detail": detail,
        })

    passed_assertions = sum(1 for a in assertion_results if a["passed"])
    total_assertions = len(assertion_results)

    result = {
        "prompt_id": prompt_id,
        "transcript_file": str(transcript_path) if transcript_path else None,
        "near_miss": prompt_data.get("near_miss", False),
        "skills_expected": skills_expected,
        "skill_metrics": skill_metrics,
        "code_blocks": {
            "total": n_code_blocks,
            "compiled": sum(compiler_results),
            "executed": sum(runner_results),
        },
        "assertions": {
            "total": total_assertions,
            "passed": passed_assertions,
            "pass_rate": round(passed_assertions / total_assertions, 4) if total_assertions > 0 else 0.0,
            "results": assertion_results,
        },
    }

    if transcript_path:
        meta = parse_transcript_filename(transcript_path)
        result["meta"] = meta

    return result


def main():
    parser = argparse.ArgumentParser(description="Check a transcript against a prompt's assertions.")
    parser.add_argument("prompt_file", type=str, help="Path to YAML prompt file")
    parser.add_argument("transcript_file", type=str, help="Path to transcript Markdown file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    prompt_data = load_yaml(args.prompt_file)
    transcript_text = load_transcript(args.transcript_file)

    # Find the specific prompt in the file
    prompt_id = Path(args.transcript_file).stem.split("--")[0]
    prompts = prompt_data.get("prompts", [])
    matched = [p for p in prompts if p["id"] == prompt_id]

    if not matched:
        print(f"Error: no prompt with id '{prompt_id}' in {args.prompt_file}", file=sys.stderr)
        sys.exit(1)

    result = run_check(matched[0], transcript_text, Path(args.transcript_file))
    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent, default=str))


if __name__ == "__main__":
    main()
