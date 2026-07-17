#!/usr/bin/env python3
"""
agents-evals runner — automated evaluation pipeline.

Drives `opencode run` for every prompt × model × mode, captures the
full JSON conversation stream, reconstructs a Markdown transcript,
and optionally runs the checker and report.

Usage:
    # Sweep all prompts across all models/modes/runs
    python agents-evals/runner.py --sweep run_config.yaml

    # Sweep + auto check + auto report
    python agents-evals/runner.py --sweep run_config.yaml --check --report

    # Single run
    python agents-evals/runner.py --prompt causality-pinn \\
        --model opencode/deepseek-v4-flash-free --mode with-skills --run 1

    # Run one full config (all prompts, one model/mode)
    python agents-evals/runner.py \\
        --model opencode/deepseek-v4-flash-free --mode with-skills

    # Check-only mode: run checker on all un-checked transcripts
    python agents-evals/runner.py --check-only
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


AGENTS_EVALS = Path(__file__).parent.resolve()
PROJECT_ROOT = AGENTS_EVALS.parent.resolve()
PROMPTS_DIR = AGENTS_EVALS / "prompts"
TRANSCRIPTS_DIR = AGENTS_EVALS / "transcripts"
RESULTS_DIR = AGENTS_EVALS / "results"
CHECKER = AGENTS_EVALS / "checker.py"
REPORT = AGENTS_EVALS / "report.py"

TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OPM = shutil.which("opencode")
if not OPM:
    print("Error: `opencode` not found on PATH.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Bare-mode state management
# ---------------------------------------------------------------------------

_BARE_PATHS = [
    (PROJECT_ROOT / "AGENTS.md", PROJECT_ROOT / "AGENTS.md.bak"),
    (PROJECT_ROOT / "CLAUDE.md", PROJECT_ROOT / "CLAUDE.md.bak"),
    (PROJECT_ROOT / ".opencode" / "skills", PROJECT_ROOT / ".opencode" / "skills.bak"),
    (PROJECT_ROOT / ".agents" / "skills", PROJECT_ROOT / ".agents" / "skills.bak"),
    (PROJECT_ROOT / ".claude" / "skills", PROJECT_ROOT / ".claude" / "skills.bak"),
]


def _enter_bare_mode():
    """Rename AGENTS.md/CLAUDE.md and skill directories so opencode can't see them."""
    moved = []
    for src, dst in _BARE_PATHS:
        if src.exists() and not dst.exists():
            os.rename(str(src), str(dst))
            moved.append((src, dst))
    return moved


def _exit_bare_mode(moved):
    """Restore renamed AGENTS.md/CLAUDE.md and skill directories."""
    for src, dst in moved:
        if dst.exists():
            os.rename(str(dst), str(src))


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_all_prompts():
    """Load every prompt from every YAML file in prompts/."""
    prompts = []
    for yaml_file in sorted(PROMPTS_DIR.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        skill = data.get("meta", {}).get("skill", yaml_file.stem)
        for p in data.get("prompts", []):
            p["_skill"] = skill
            p["_yaml_file"] = str(yaml_file)
            prompts.append(p)
    return prompts


def find_prompt(prompt_id):
    """Find a single prompt by ID."""
    for p in load_all_prompts():
        if p["id"] == prompt_id:
            return p
    return None


# ---------------------------------------------------------------------------
# JSONL -> Markdown transcript
# ---------------------------------------------------------------------------


def reconstruct_transcript(json_lines, prompt_text, model_provider, mode,
                           wall_time=None):
    """Rebuild a Markdown transcript from opencode's JSONL event stream.

    Returns (transcript_string, meta_dict) where meta_dict contains:
      - tokens: aggregated token counts
      - wall_time: elapsed seconds (if provided)
      - tool_counts: dict of tool_name -> count
      - skills_loaded: list of skill names
      - step_count: number of steps
    """
    lines = []
    lines.append(f"# {prompt_text[:80]} — {model_provider} — {mode}")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Prompt**: {prompt_text}")
    lines.append(f"- **Model**: {model_provider}")
    lines.append(f"- **Mode**: {mode}")
    if wall_time is not None:
        lines.append(f"- **Wall time**: {wall_time:.1f}s")
    lines.append("")

    # Aggregators
    tokens_agg = {"total": 0, "input": 0, "output": 0, "reasoning": 0,
                  "cache_write": 0, "cache_read": 0}
    tool_summary = []
    skill_names_loaded = []

    for raw_line in json_lines:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "?")
        part = event.get("part", {})

        if etype == "__turn__":
            turn_num = event.get("turn", 0)
            label = event.get("label", f"Turn {turn_num}")
            lines.append("")
            lines.append(f"--- **{label}** ---")
            lines.append("")
            lines.append("")

        elif etype == "step_start":
            lines.append("---")
            lines.append("")

        elif etype == "text":
            text = part.get("text", "")
            if text.strip():
                lines.append(text)
                lines.append("")

        elif etype == "tool_use":
            tool_name = part.get("tool", "?")
            call_id = part.get("callID", "?")
            state = part.get("state", {})
            status = state.get("status", "?")
            inp = state.get("input", {})
            out = state.get("output", {})

            entry = {"tool": tool_name, "call_id": call_id, "status": status}
            tool_summary.append(entry)

            if tool_name == "skill":
                skill_names_loaded.append(inp.get("name", "?"))

            lines.append(f"> **Tool: {tool_name}** (`{call_id}`) — {status}")
            if inp:
                inp_str = json.dumps(inp, indent=2, default=str)
                lines.append("> ```json")
                for il in inp_str.split("\n"):
                    lines.append(f"> {il}")
                lines.append("> ```")
            if out:
                out_str = json.dumps(out, indent=2, default=str)[:500]
                lines.append("> *Output:*")
                for ol in out_str.split("\n"):
                    lines.append(f">   {ol}")
            lines.append("")

        elif etype == "step_finish":
            reason = part.get("reason", "?")
            step_tokens = part.get("tokens", {})
            for k in tokens_agg:
                tokens_agg[k] += step_tokens.get(k, 0)
            lines.append(f"*Step finished — reason: {reason}*")
            if step_tokens:
                lines.append(f"*Tokens — total: {step_tokens.get('total', '?')}, "
                             f"output: {step_tokens.get('output', '?')}*")
            lines.append("")

    # Append token summary if any
    if tokens_agg["total"] > 0:
        lines.append("")
        lines.append("## Token usage")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total input tokens | {tokens_agg['input']} |")
        lines.append(f"| Total output tokens | {tokens_agg['output']} |")
        lines.append(f"| Reasoning tokens | {tokens_agg['reasoning']} |")
        lines.append(f"| Cache write | {tokens_agg['cache_write']} |")
        lines.append(f"| Cache read | {tokens_agg['cache_read']} |")
        lines.append(f"| **Grand total** | **{tokens_agg['total']}** |")
        lines.append("")

    # Append tool summary table at the end
    if tool_summary:
        lines.append("")
        lines.append("## Tool call summary")
        lines.append("")
        lines.append("| # | Tool | Status |")
        lines.append("|---|------|--------|")
        for i, entry in enumerate(tool_summary, 1):
            lines.append(f"| {i} | {entry['tool']} | {entry['status']} |")
        lines.append("")

    # Count tools by name
    tool_counts = {}
    for entry in tool_summary:
        t = entry["tool"]
        tool_counts[t] = tool_counts.get(t, 0) + 1

    meta = {
        "tokens": tokens_agg,
        "wall_time": wall_time,
        "tool_counts": tool_counts,
        "skills_loaded": skill_names_loaded,
        "step_count": len([e for e in json_lines if json.loads(e).get("type") == "step_start"]),
    }
    return "\n".join(lines), meta


# ---------------------------------------------------------------------------
# Run opencode
# ---------------------------------------------------------------------------


def run_opencode(prompt_text, model_provider, mode, timeout_seconds=120,
                 continue_session=False, verbose=False):
    """Execute `opencode run --format json` and return parsed lines + raw bytes.

    If continue_session=True, uses --continue to continue the last session.
    If verbose=True, prints the model's text responses to stderr live.
    """
    cmd = [
        OPM, "run",
        "--format", "json",
        "--auto",
        "--model", model_provider,
    ]
    if continue_session:
        cmd.append("--continue")
    cmd.append(prompt_text)

    # Bare mode: hide AGENTS.md and skill directories
    moved = []
    if mode == "bare" and not continue_session:
        moved = _enter_bare_mode()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            cwd=str(PROJECT_ROOT),
        )

        stdout_data = []
        while True:
            try:
                line = proc.stdout.readline()
            except Exception:
                break
            if not line and proc.poll() is not None:
                break
            if line:
                stdout_data.append(line)
                if verbose:
                    try:
                        event = json.loads(line.decode("utf-8", errors="replace").strip())
                        if event.get("type") == "text":
                            text = event.get("part", {}).get("text", "")
                            if text.strip():
                                print(text, file=sys.stderr, flush=True)
                    except json.JSONDecodeError:
                        pass

        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        stderr_data = proc.stderr.read() if proc.stderr else b""

    finally:
        if mode == "bare" and not continue_session:
            _exit_bare_mode(moved)

    raw = b"".join(stdout_data)
    # Parse JSONL — each line is a JSON object
    json_lines = []
    for line in raw.decode("utf-8", errors="replace").split("\n"):
        line = line.strip()
        if line:
            json_lines.append(line)

    return json_lines, raw, stderr_data


# ---------------------------------------------------------------------------
# Transcript filename
# ---------------------------------------------------------------------------


def transcript_filename(prompt_id, model_id, mode, run_num):
    """Generate a filename like `causality-pinn--deepseek-v4--with-skills--run1.md`."""
    safe_model = model_id.replace("/", "-").replace(" ", "-")
    return f"{prompt_id}--{safe_model}--{mode}--run{run_num}.md"


# ---------------------------------------------------------------------------
# Checker integration
# ---------------------------------------------------------------------------


def run_checker(prompt_yaml, transcript_path):
    """Run checker.py and return the result dict."""
    result = subprocess.run(
        [sys.executable, str(CHECKER), prompt_yaml, str(transcript_path)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip()}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": f"invalid JSON from checker: {result.stdout[:200]}"}


def run_report():
    """Run report.py and print its output."""
    result = subprocess.run(
        [sys.executable, str(REPORT), "--results", str(RESULTS_DIR), "--all"],
        capture_output=True, text=True, timeout=30,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500], file=sys.stderr)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_single(prompt, model_provider, model_id, mode, run_num,
               run_check=True, timeout=120, verbose=False):
    """Run one prompt through opencode, save transcript, optionally check.

    Supports multi-turn: if the prompt has a ``turns`` list, each turn is
    sent as a follow-up via ``--continue`` and the results are combined
    into a single transcript.
    """
    prompt_id = prompt["id"]
    prompt_text = prompt["text"]
    prompt_yaml = prompt["_yaml_file"]
    turns = prompt.get("turns", [])
    max_turns = prompt.get("max_turns", len(turns))
    filename = transcript_filename(prompt_id, model_id, mode, run_num)
    transcript_path = TRANSCRIPTS_DIR / filename
    result_path = RESULTS_DIR / filename.replace(".md", ".json")

    if transcript_path.exists():
        print(f"  ⏭️  Already exists: {filename}")
        if run_check and not result_path.exists():
            res = run_checker(prompt_yaml, transcript_path)
            with open(result_path, "w") as f:
                json.dump(res, f, indent=2, default=str)
            a = res.get("assertions", {})
            print(f"     Checked: {a.get('passed', 0)}/{a.get('total', 0)} passed")
        return

    print(f"  🚀 Running: {filename}")
    sys.stdout.flush()

    # --- Turn 1: send the main prompt ---
    t_start = time.time()
    all_json_lines = []
    turn_count = 0

    json_lines, raw, stderr = run_opencode(prompt_text, model_provider, mode, timeout, verbose=verbose)
    if json_lines:
        all_json_lines.append({"turn": 0, "label": "prompt", "lines": json_lines})
        turn_count = 1

    # --- Subsequent turns: continue the session with pre-answers ---
    turn_timeout = max(timeout // (max_turns + 1), 30)
    for i, answer in enumerate(turns[:max_turns]):
        if verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"  [VERBOSE] Turn {i+1}: {answer}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
        else:
            print(f"     ↳ Turn {i+1}: {answer[:80]}...")
        sys.stdout.flush()
        t_json, t_raw, t_stderr = run_opencode(
            answer, model_provider, mode, turn_timeout, continue_session=True, verbose=verbose
        )
        if t_json:
            all_json_lines.append({"turn": i + 1, "label": f"answer: {answer[:60]}", "lines": t_json})
            turn_count += 1

    wall_time = time.time() - t_start

    # Combine all turns into a single JSON lines list for transcript
    combined_lines = []
    for turn_group in all_json_lines:
        if turn_group["turn"] > 0:
            combined_lines.append(
                json.dumps({"type": "__turn__", "turn": turn_group["turn"],
                            "label": turn_group["label"]})
            )
        combined_lines.extend(turn_group["lines"])

    if not combined_lines:
        print(f"  ⚠️  No output (timeout?). Saving stderr.", file=sys.stderr)
        with open(transcript_path, "w") as f:
            f.write(f"# {prompt_text[:80]}\n\nNo JSON output received.\n\n")
            for tg in all_json_lines:
                f.write(f"--- Turn {tg['turn']} ---\n")
            f.write(f"\nstderr:\n{stderr.decode(errors='replace')}\n")
        return

    n_turns_label = f" (multi-turn: {turn_count} turns)" if turn_count > 1 else ""
    transcript, meta = reconstruct_transcript(
        combined_lines, prompt_text, model_provider, mode, wall_time=wall_time
    )
    if turn_count > 1:
        meta["turn_count"] = turn_count
    with open(transcript_path, "w") as f:
        f.write(transcript)

    print(f"     Saved: {filename} ({len(combined_lines)} events{n_turns_label}, "
          f"{meta['tokens']['total']} tokens, {wall_time:.0f}s)")
    sys.stdout.flush()

    if run_check:
        res = run_checker(prompt_yaml, transcript_path)
        res["runtime"] = meta
        with open(result_path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        a = res.get("assertions", {})
        sm = res.get("skill_metrics", {})
        to = meta.get("tokens", {})
        f1_val = sm.get('f1')
        f1_str = f"{f1_val:.2f}" if isinstance(f1_val, (int, float)) else "N/A"
        print(f"     Checked: {a.get('passed', 0)}/{a.get('total', 0)} passed, "
              f"F1={f1_str}, tokens={to.get('total', 0)}")
        sys.stdout.flush()


def run_all(model_provider, model_id, mode, run_num=None,
            prompt_filter=None, run_check=True, timeout=120, verbose=False):
    """Run all prompts for one model/mode combination."""
    prompts = load_all_prompts()
    if prompt_filter:
        prompts = [p for p in prompts if p["id"] in prompt_filter]

    runs_to_do = [run_num] if run_num else [1]
    total = len(prompts) * len(runs_to_do)
    done = 0

    for p in prompts:
        near_miss = p.get("near_miss", False)
        label = "[NEAR-MISS] " if near_miss else ""
        print(f"\n  [{done+1}/{total}] {label}{p['id']} ({p['_skill']})")
        for rn in runs_to_do:
            run_single(p, model_provider, model_id, mode, rn,
                       run_check=run_check, timeout=timeout, verbose=verbose)
            done += 1


def sweep(config_path, run_check=True, run_report_flag=False, timeout=120, verbose=False):
    """Full sweep: all models × modes × runs."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    models = config.get("models", [])
    modes = config.get("modes", ["with-skills"])
    runs = config.get("runs", 1)

    total = (
        len(models)
        * len(modes)
        * runs
        * len(load_all_prompts())
    )
    print(f"Full sweep: {len(models)} models × {len(modes)} modes × {runs} runs "
          f"× {len(load_all_prompts())} prompts = {total} evaluations")
    print()

    for model_cfg in models:
        model_id = model_cfg["id"]
        provider = model_cfg["provider"]
        for mode in modes:
            print(f"\n{'='*60}")
            print(f"Model: {model_id}  |  Mode: {mode}")
            print(f"{'='*60}")
            for rn in range(1, runs + 1):
                run_all(provider, model_id, mode, run_num=rn,
                        run_check=run_check, timeout=timeout, verbose=verbose)

    print(f"\n{'='*60}")
    print("Sweep complete!")
    print(f"{'='*60}")

    if run_report_flag:
        print("\nGenerating report...")
        run_report()


def check_only():
    """Run checker on all transcripts that don't have a result yet."""
    count = 0
    for transcript_path in sorted(TRANSCRIPTS_DIR.glob("*.md")):
        filename = transcript_path.name
        result_path = RESULTS_DIR / filename.replace(".md", ".json")
        if result_path.exists():
            continue

        # Find the matching prompt YAML
        prompt_id = filename.split("--")[0]
        prompt_yaml = None
        for yaml_file in PROMPTS_DIR.glob("*.yaml"):
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            for p in data.get("prompts", []):
                if p["id"] == prompt_id:
                    prompt_yaml = yaml_file
                    break
            if prompt_yaml:
                break

        if not prompt_yaml:
            print(f"  ⚠️  No prompt file found for {filename}")
            continue

        print(f"  🔍 Checking: {filename}")
        res = run_checker(str(prompt_yaml), transcript_path)
        with open(result_path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        a = res.get("assertions", {})
        print(f"     {a.get('passed', 0)}/{a.get('total', 0)} passed")
        count += 1

    print(f"\n  Checked {count} transcript(s). Total results: {len(list(RESULTS_DIR.glob('*.json')))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Automated PINA skill evaluation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sweep", type=str, help="Path to sweep config YAML")
    parser.add_argument("--prompt", type=str, help="Single prompt ID to run")
    parser.add_argument("--model", type=str, help="Model provider string (e.g. opencode/deepseek-v4-flash-free)")
    parser.add_argument("--model-id", type=str, help="Short model ID for filenames (defaults from --model)")
    parser.add_argument("--mode", type=str, choices=["with-skills", "bare"], default="with-skills")
    parser.add_argument("--run", type=int, default=1, help="Run number (default: 1)")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1, only used with --model)")
    parser.add_argument("--check", action="store_true", help="Run checker after each evaluation")
    parser.add_argument("--report", action="store_true", help="Run report after sweep")
    parser.add_argument("--check-only", action="store_true", help="Run checker on all unchecked transcripts")
    parser.add_argument("--timeout", type=int, default=120, help="Per-prompt timeout in seconds (default: 120)")
    parser.add_argument("--verbose", action="store_true", help="Print model responses live to stderr")

    args = parser.parse_args()

    if args.check_only:
        check_only()
        return

    if args.sweep:
        sweep(args.sweep, run_check=args.check, run_report_flag=args.report,
              timeout=args.timeout, verbose=args.verbose)
        return

    if not args.model:
        parser.print_help()
        print("\nError: --model or --sweep required", file=sys.stderr)
        sys.exit(1)

    # Derive model_id from provider if not given
    model_id = args.model_id or args.model.split("/")[-1]

    if args.prompt:
        p = find_prompt(args.prompt)
        if not p:
            print(f"Error: prompt '{args.prompt}' not found", file=sys.stderr)
            sys.exit(1)
        run_single(p, args.model, model_id, args.mode, args.run,
                   run_check=args.check, timeout=args.timeout, verbose=args.verbose)
    else:
        for rn in range(1, args.runs + 1):
            run_all(args.model, model_id, args.mode, run_num=rn,
                    run_check=args.check, timeout=args.timeout, verbose=args.verbose)


if __name__ == "__main__":
    main()
