#!/usr/bin/env python3
"""
agents-evals report — aggregation, comparison, and human-friendly metrics.

Usage:
    python report.py --results <dir>              # aggregate all check results
    python report.py --results <dir> --by-model   # agent comparison matrix
    python report.py --results <dir> --by-skill   # per-skill breakdown
    python report.py --results <dir> --variance   # mean ± std across runs
    python report.py --results <dir> --all        # everything
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def load_results(dir_path):
    """Load all JSON result files from a directory."""
    results = []
    for f in sorted(Path(dir_path).glob("*.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def bar(value, width=20, max_val=1.0, char="█"):
    """Generate a text bar."""
    filled = int((value / max_val) * width) if max_val > 0 else 0
    return char * filled + "░" * (width - filled)


def print_section(title):
    """Print a section header."""
    print()
    print(title)
    print("━" * len(title))


def aggregate_by_skill(results):
    """Group results by skill."""
    # Map prompt_ids to skills via lookup from YAML files
    # We extract from the results metadata
    skill_map = {}
    prompt_dir = Path(__file__).parent / "prompts"
    try:
        import yaml
        for yaml_file in prompt_dir.glob("*.yaml"):
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            skill = data.get("meta", {}).get("skill", yaml_file.stem)
            for p in data.get("prompts", []):
                skill_map[p["id"]] = skill
    except Exception:
        # Fallback: use prompt_id prefix before first number
        for r in results:
            pid = r.get("prompt_id", "unknown")
            skill_map[pid] = pid.rsplit("-", 1)[0] if "-" in pid else pid
    except ImportError:
        pass

    by_skill = defaultdict(list)
    for r in results:
        pid = r.get("prompt_id", "unknown")
        skill = skill_map.get(pid, "unknown")
        by_skill[skill].append(r)
    return by_skill


def aggregate_by_model(results):
    """Group results by model."""
    by_model = defaultdict(list)
    for r in results:
        model = r.get("meta", {}).get("model", "unknown")
        by_model[model].append(r)
    return by_model


def aggregate_by_model_skills(results):
    """Group by (model, skills_mode)."""
    groups = defaultdict(list)
    for r in results:
        meta = r.get("meta", {})
        key = (meta.get("model", "unknown"), meta.get("skills_mode", "unknown"))
        groups[key].append(r)
    return groups


def compute_avg_metric(results, metric_key):
    """Compute average of a nested metric across results."""
    values = []
    for r in results:
        # Support dot-notation keys like "skill_metrics.f1"
        parts = metric_key.split(".")
        val = r
        try:
            for p in parts:
                val = val[p]
            if val is not None:
                values.append(val)
        except (KeyError, TypeError):
            continue
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    return mean, std


def compute_pass_rate(results):
    """Compute overall assertion pass rate across results."""
    total = 0
    passed = 0
    for r in results:
        a = r.get("assertions", {})
        total += a.get("total", 0)
        passed += a.get("passed", 0)
    if total == 0:
        return 0.0
    return passed / total


def compute_code_rate(results, key):
    """Compute code compile or execute rate."""
    total = 0
    success = 0
    for r in results:
        cb = r.get("code_blocks", {})
        total += cb.get("total", 0)
        success += cb.get(key, 0)
    if total == 0:
        return 0.0
    return success / total


def compute_avg_runtime(results, key):
    """Compute average of a runtime metric (tokens, wall_time, etc.)."""
    values = []
    for r in results:
        rm = r.get("runtime", {})
        if key == "tokens_total":
            val = rm.get("tokens", {}).get("total", 0)
        elif key == "tokens_input":
            val = rm.get("tokens", {}).get("input", 0)
        elif key == "tokens_output":
            val = rm.get("tokens", {}).get("output", 0)
        elif key == "wall_time":
            val = rm.get("wall_time", 0)
        else:
            val = 0
        if val:
            values.append(val)
    if not values:
        return 0.0
    return sum(values) / len(values)


def report_skill_summary(results):
    """Print per-skill summary table."""
    by_skill = aggregate_by_skill(results)
    print_section("Per-Skill Summary — Precision / Recall / F1")
    h = f"{'Skill':<22} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Pass':>6} {'Comp':>6} {'Exec':>6} {'Tokens':>8} {'N':>4}"
    print(h)
    print("-" * (len(h) + 4))

    for skill in sorted(by_skill.keys()):
        res = by_skill[skill]
        p_mean, _ = compute_avg_metric(res, "skill_metrics.precision")
        r_mean, _ = compute_avg_metric(res, "skill_metrics.recall")
        f_mean, _ = compute_avg_metric(res, "skill_metrics.f1")
        pass_rate = compute_pass_rate(res)
        comp_rate = compute_code_rate(res, "compiled")
        exec_rate = compute_code_rate(res, "executed")
        avg_tokens = compute_avg_runtime(res, "tokens_total")
        n = len(res)

        p_bar = bar(p_mean, 12)
        label = f"{skill:<22}"
        print(f"{label} {p_mean:>5.2f} {r_mean:>5.2f} {f_mean:>5.2f} {pass_rate:>5.2f} {comp_rate:>5.2f} {exec_rate:>5.2f} {avg_tokens:>7.0f} {n:>4}  {p_bar}")


def report_model_comparison(results):
    """Print agent/model comparison matrix."""
    groups = aggregate_by_model_skills(results)

    print_section("Agent Comparison — F1 × Pass Rate × Tokens × Time")
    h = f"{'Agent':<32} {'F1':>6} {'Pass':>6} {'Exec':>6} {'Comp':>6} {'Tokens':>8} {'Time':>6} {'N':>4}"
    print(h)
    print("-" * (len(h) + 4))

    for key in sorted(groups.keys(), key=lambda k: (k[0], k[1])):
        model, mode = key
        res = groups[key]
        f_mean, _ = compute_avg_metric(res, "skill_metrics.f1")
        pass_rate = compute_pass_rate(res)
        exec_rate = compute_code_rate(res, "executed")
        comp_rate = compute_code_rate(res, "compiled")
        avg_tokens = compute_avg_runtime(res, "tokens_total")
        avg_time = compute_avg_runtime(res, "wall_time")
        n = len(res)

        label = f"{model:<20} {mode:<10}"
        print(f"{label} {f_mean:>5.2f} {pass_rate:>5.2f} {exec_rate:>5.2f} {comp_rate:>5.2f} {avg_tokens:>7.0f} {avg_time:>5.0f}s {n:>4}")

    print()
    print("F1 × Pass Rate chart:")
    for key in sorted(groups.keys(), key=lambda k: (k[0], k[1])):
        model, mode = key
        res = groups[key]
        f_mean, _ = compute_avg_metric(res, "skill_metrics.f1")
        pass_rate = compute_pass_rate(res)
        composite = f_mean * pass_rate
        label = f"{model:<20} {mode:<10}"
        print(f"{label} {bar(composite, 30)}  {composite:.3f}")


def report_model_delta(results):
    """Compare with-skills vs bare for each model."""
    groups = aggregate_by_model_skills(results)

    model_modes = defaultdict(dict)
    for (model, mode), res in groups.items():
        f_mean, _ = compute_avg_metric(res, "skill_metrics.f1")
        pass_rate = compute_pass_rate(res)
        comp_rate = compute_code_rate(res, "compiled")
        exec_rate = compute_code_rate(res, "executed")
        avg_tokens = compute_avg_runtime(res, "tokens_total")
        avg_time = compute_avg_runtime(res, "wall_time")
        model_modes[model][mode] = {
            "f1": f_mean, "pass_rate": pass_rate,
            "compiled": comp_rate, "executed": exec_rate,
            "tokens": avg_tokens, "time": avg_time,
        }

    print_section("With-Skills vs Bare Delta (skill value-add)")
    h = f"{'Model':<22} {'ΔF1':>8} {'ΔPass':>8} {'ΔExec':>8} {'ΔComp':>8} {'ΔTokens':>9} {'ΔTime':>8}"
    print(h)
    print("-" * (len(h) + 4))

    for model in sorted(model_modes.keys()):
        modes = model_modes[model]
        if "with-skills" in modes and "bare" in modes:
            ws = modes["with-skills"]
            bs = modes["bare"]
            df1 = ws["f1"] - bs["f1"]
            dp = ws["pass_rate"] - bs["pass_rate"]
            de = ws["executed"] - bs["executed"]
            dc = ws["compiled"] - bs["compiled"]
            dt = ws["tokens"] - bs["tokens"]
            dtime = ws["time"] - bs["time"]
            print(f"{model:<22} {df1:>+7.2f} {dp:>+7.2f} {de:>+7.2f} {dc:>+7.2f} {dt:>+8.0f} {dtime:>+7.0f}s")
        else:
            modes_str = ", ".join(modes.keys())
            print(f"{model:<22} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>9} {'—':>8}  (modes: {modes_str})")


def report_near_miss(results):
    """Report near-miss accuracy."""
    near_miss_results = [r for r in results if r.get("near_miss", False)]
    normal_results = [r for r in results if not r.get("near_miss", False)]

    if not near_miss_results:
        print("\nNo near-miss prompts evaluated.")
        return

    correct = sum(1 for r in near_miss_results if r.get("skill_metrics", {}).get("precision", 0) == 1.0)
    total = len(near_miss_results)

    print_section("Near-Miss Accuracy (should NOT trigger any skill)")
    for r in near_miss_results:
        pid = r.get("prompt_id", "?")
        sm = r.get("skill_metrics", {})
        fp = sm.get("false_positives", 0)
        unexpected = sm.get("unexpected", [])
        status = "✅" if fp == 0 else "❌"
        trigger_info = f" (false positives: {unexpected})" if unexpected else ""
        print(f"  {status} {pid}{trigger_info}")

    print(f"\n  Accuracy: {correct}/{total} ({correct / total * 100:.0f}%)")


def report_variance(results):
    """Report mean ± std across runs for each prompt that has multiple runs."""
    by_prompt = defaultdict(list)
    for r in results:
        pid = r.get("prompt_id", "?")
        by_prompt[pid].append(r)

    print_section("Variance Across Runs (mean ± std)")
    print(f"{'Prompt':<30} {'F1':>12} {'Pass Rate':>12} {'Compile':>10} {'Runs':>6}")
    print("-" * 70)

    for pid in sorted(by_prompt.keys()):
        res = by_prompt[pid]
        if len(res) < 2:
            continue

        f1_vals = [r.get("skill_metrics", {}).get("f1", 0) or 0 for r in res]
        pass_vals = [r.get("assertions", {}).get("pass_rate", 0) for r in res]
        comp_vals = []
        for r in res:
            cb = r.get("code_blocks", {})
            c = cb.get("compiled", 0)
            t = cb.get("total", 1)
            comp_vals.append(c / t if t > 0 else 0)

        def stats(vals):
            m = sum(vals) / len(vals)
            v = sum((x - m) ** 2 for x in vals) / len(vals)
            s = v ** 0.5
            return m, s

        f1_m, f1_s = stats(f1_vals)
        pa_m, pa_s = stats(pass_vals)
        co_m, co_s = stats(comp_vals)

        print(f"{pid:<30} {f1_m:.2f}±{f1_s:.2f}  {pa_m:.2f}±{pa_s:.2f}  {co_m:.2f}±{co_s:.2f}  {len(res):>3}")


def report_overall(results):
    """Print overall summary."""
    total = len(results)
    with_code = sum(1 for r in results if r.get("code_blocks", {}).get("total", 0) > 0)
    compiled_all = sum(
        1 for r in results
        if r.get("code_blocks", {}).get("total", 0) > 0
        and r["code_blocks"]["compiled"] == r["code_blocks"]["total"]
    )
    executed_all = sum(
        1 for r in results
        if r.get("code_blocks", {}).get("total", 0) > 0
        and r["code_blocks"]["executed"] == r["code_blocks"]["total"]
    )
    pass_rate = compute_pass_rate(results)
    comp_rate = compute_code_rate(results, "compiled")
    exec_rate = compute_code_rate(results, "executed")
    avg_tokens = compute_avg_runtime(results, "tokens_total")
    avg_time = compute_avg_runtime(results, "wall_time")

    print_section("Overall Summary")
    print(f"  Total prompts evaluated:     {total}")
    print(f"  Prompts with code blocks:    {with_code}")
    print(f"  All code compiled:           {compiled_all}/{with_code}")
    print(f"  All code executed:           {executed_all}/{with_code}")
    print(f"  Overall assertion pass rate: {pass_rate:.2%}")
    print(f"  Code compilation rate:       {comp_rate:.2%}")
    print(f"  Code execution rate:         {exec_rate:.2%}")
    print(f"  Average tokens per prompt:   {avg_tokens:.0f}")
    print(f"  Average wall time per prompt: {avg_time:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation reports from checker results.")
    parser.add_argument("--results", "-r", type=str, default="results",
                        help="Directory containing checker JSON result files")
    parser.add_argument("--all", "-a", action="store_true", help="Print all reports")
    parser.add_argument("--by-skill", action="store_true", help="Per-skill breakdown")
    parser.add_argument("--by-model", action="store_true", help="Agent comparison matrix")
    parser.add_argument("--delta", action="store_true", help="With-skills vs bare delta")
    parser.add_argument("--variance", action="store_true", help="Run variance (requires multiple runs)")
    parser.add_argument("--near-miss", action="store_true", help="Near-miss accuracy")
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"Error: results directory '{args.results}' not found", file=sys.stderr)
        sys.exit(1)

    results = load_results(results_dir)
    if not results:
        print(f"Error: no JSON results found in '{args.results}'", file=sys.stderr)
        sys.exit(1)

    do_all = args.all

    report_overall(results)

    if do_all or args.by_skill:
        report_skill_summary(results)

    if do_all or args.by_model:
        report_model_comparison(results)

    if do_all or args.delta:
        report_model_delta(results)

    if do_all or args.variance:
        report_variance(results)

    if do_all or args.near_miss:
        report_near_miss(results)

    print()


if __name__ == "__main__":
    main()
