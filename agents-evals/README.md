# agents-evals — PINA Skill Evaluation Pipeline

Systematic testing for PINA agent skills. Test whether skills trigger correctly,
generate compilable/runnable code, and add value over a bare model.

## Directory layout

```
agents-evals/
├── README.md                 ← you are here
├── prompts/                  ← 11 YAML files (42 prompts + assertions)
├── transcripts/              ← save full conversations here
├── results/                  ← checker JSON outputs (auto-generated)
├── checker.py                ← assertion engine
├── report.py                 ← aggregation and metrics
├── runner.py                 ← automated evaluation pipeline
├── run_config.yaml           ← sweep configuration (models × modes × runs)
├── run_eval.sh               ← lifecycle script
└── template_transcript.md    ← filename convention reminder
```

## Prompt catalog

42 prompts across 11 categories. Each prompt targets specific skills with
checkable assertions. Prompts with `turns` have pre-configured follow-up
answers for multi-turn conversations.

| Prompt ID | Skill / Category | Skills expected |
|-----------|-----------------|-----------------|
| `navier-stokes-sensor` | pina-workflow | pina-workflow, create-problem, define-equations, define-domains, condition-setup, select-model, select-solver, select-trainer |
| `damped-pendulum-inverse` | pina-workflow | pina-workflow, create-problem, define-equations, condition-setup, select-model, select-solver, select-trainer |
| `schrodinger-1d` | pina-workflow | pina-workflow, create-problem, define-equations, define-domains, condition-setup, select-model, select-solver, select-trainer |
| `weather-graph-gnn` | pina-workflow | pina-workflow, create-problem, condition-setup, select-model, select-solver, select-trainer |
| `diffusion-parametric` | create-problem | create-problem, define-domains, define-equations, condition-setup |
| `lorenz-ode` | create-problem | create-problem, define-equations, condition-setup |
| `data-driven-material` | create-problem | create-problem, condition-setup |
| `wave-plus-sensors` | create-problem | create-problem, define-domains, define-equations, condition-setup |
| `l-shaped-region` | define-domains | define-domains |
| `annulus` | define-domains | define-domains |
| `triangular-domain` | define-domains | define-domains |
| `ellipsoid-lhs` | define-domains | define-domains |
| `navier-stokes-custom` | define-equations | define-equations |
| `robin-boundary` | define-equations | define-equations |
| `inverse-reaction-diffusion` | define-equations | define-equations |
| `kdv-equation` | define-equations | define-equations |
| `graph-time-series` | condition-setup | condition-setup |
| `multi-condition-mix` | condition-setup | condition-setup |
| `pyg-molecule` | condition-setup | condition-setup |
| `unstructured-cfd-gno` | select-model | select-model |
| `sindy-discovery` | select-model | select-model |
| `multi-scale-boundary` | select-model | select-model |
| `few-shot-regression` | select-model | select-model |
| `causality-pinn` | select-solver | select-solver |
| `ensemble-uncertainty` | select-solver | select-solver |
| `sharp-interfaces` | select-solver | select-solver |
| `adversarial-pinn` | select-solver | select-solver |
| `imbalanced-batching` | select-trainer | select-trainer |
| `mixed-precision-early-stop` | select-trainer | select-trainer |
| `mixed-condition-batching` | select-trainer | select-trainer |
| `viz-skill` | create-skill | create-skill |
| `multi-gpu-skill` | create-skill | create-skill |
| `nan-debug-skill` | create-skill | create-skill |
| `sync-select-solver` | skill-sync-checker | skill-sync-checker |
| `stiff-odes-methods` | near-miss | *(should NOT trigger any skill)* |
| `heat-equation-uniqueness` | near-miss | *(should NOT trigger any skill)* |
| `cnn-vs-transformer` | near-miss | *(should NOT trigger any skill)* |
| `adam-vs-sgd` | near-miss | *(should NOT trigger any skill)* |
| `close-but-wrong-phrase` | near-miss | *(should NOT trigger any skill)* |
| `run-tests` | near-miss | *(should NOT trigger any skill)* |
| `what-is-tensor` | near-miss | *(should NOT trigger any skill)* |
| `setup-venv` | near-miss | *(should NOT trigger any skill)* |

Run a single prompt:

```bash
python agents-evals/runner.py --prompt causality-pinn --model opencode/deepseek-v4-flash-free --check
```

## Prerequisites

- **Python 3.8+** with PyYAML (`pip install pyyaml`)
- **[opencode](https://opencode.ai) CLI** installed and available on `PATH`
  - Used to run prompts automatically against different models
  - Tested with opencode v1.17+ (run `opencode --version` to check)
- The PINA skills must be present in this repo's `.opencode/skills/`,
  `.agents/skills/`, and `.claude/skills/` directories (they already are if
  this is the PINA repo)

### Installing opencode

See [opencode.ai](https://opencode.ai) for installation instructions. On macOS:

```bash
curl -fsSL https://opencode.ai/install.sh | sh
```

Then verify:

```bash
opencode --version
opencode models
```

### Available models

List models available through your opencode provider:

```bash
opencode models
```

Eval defaults reference these model IDs in `run_config.yaml`:
- `opencode/deepseek-v4-flash-free`
- `opencode/hy3-free`
- `opencode/mimo-v2.5-free`

You can also run against any model opencode supports by passing `--model`
directly.

## Two ways to run

### Option A: Fully automated (uses opencode CLI)

The `runner.py` script drives `opencode run --format json` for every prompt
pairing, auto-checking each result and generating reports.

```bash
# Run one prompt
python agents-evals/runner.py \
  --prompt causality-pinn \
  --model opencode/deepseek-v4-flash-free \
  --mode with-skills --run 1 --check

# Run all prompts for one model/mode
python agents-evals/runner.py \
  --model opencode/deepseek-v4-flash-free \
  --mode with-skills --check

# Full sweep: all models × modes × runs
python agents-evals/runner.py --sweep agents-evals/run_config.yaml --check --report
```

Or equivalently through the shell wrapper:

```bash
bash run_eval.sh run --prompt causality-pinn --model opencode/deepseek-v4-flash-free --mode with-skills
bash run_eval.sh run --sweep agents-evals/run_config.yaml --check --report
```

**Bare mode**: `runner.py` temporarily renames AGENTS.md/CLAUDE.md and moves
the skill directories before running, then restores them. This prevents
opencode from loading any skills, giving you a clean baseline comparison.

### Option B: Manual (collect transcripts yourself)

1. Pick a prompt from `prompts/*.yaml`
2. Paste it into any agent (Claude Code, opencode TUI, Codex CLI)
3. Save the full conversation to `transcripts/<prompt-id>--<model>--<mode>--<run>.md`
4. Run the checker:

```bash
bash run_eval.sh check prompts/<yaml-file>.yaml transcripts/<transcript>.md
```

## Quick start (automated)

```bash
# 1. List all prompts
bash agents-evals/run_eval.sh next

# 2. Run one prompt through opencode automatically
python agents-evals/runner.py \
  --prompt causality-pinn \
  --model opencode/deepseek-v4-flash-free \
  --run 1 --check

# 3. Run all prompts for one model
python agents-evals/runner.py \
  --model opencode/deepseek-v4-flash-free \
  --mode with-skills --check

# 4. Compare: run the same model bare
python agents-evals/runner.py \
  --model opencode/deepseek-v4-flash-free \
  --mode bare --check

# 5. Generate report
python agents-evals/report.py --results agents-evals/results --all
```

## Running the full eval cycle

### Phase 1 — Collect transcripts

**Automated** (recommended):

```bash
# Full sweep with 3 runs for variance
python agents-evals/runner.py --sweep agents-evals/run_config.yaml --check
```

This will loop through every prompt × model × mode × run, saving transcripts
and checker results automatically.

**Manual** (for testing models outside opencode, e.g. Claude Code):

For each prompt in `prompts/*.yaml`, paste it into your chosen model with or
without PINA skills loaded. Save the full conversation to `transcripts/`.

**Filename convention:**
```
<prompt-id>--<model>--(with-skills|bare)--<run>.md
```

Examples:
- `causality-pinn--deepseek-v4--with-skills--run1.md`
- `navier-stokes-sensor--claude-opus-4--bare--run1.md`
- `l-shaped-region--fable--with-skills--run3.md`

For variance: repeat each (prompt × model × mode) 3 times with `run1`, `run2`, `run3`.

### Phase 2 — Batch check

```bash
bash agents-evals/run_eval.sh check-batch
# Or equivalently:
python agents-evals/runner.py --check-only
```

This runs the checker on every transcript that doesn't have a corresponding
result yet. Results are saved as JSON to `results/`.

### Phase 3 — Generate reports

```bash
# Overall stats
bash agents-evals/run_eval.sh report

# Per-skill precision/recall/F1
bash agents-evals/run_eval.sh report --by-skill

# Agent comparison matrix (with-skills vs bare across models)
bash agents-evals/run_eval.sh report --by-model

# Skill value-add delta (how much skills improve each model)
bash agents-evals/run_eval.sh report --delta

# Run variance (requires 3 runs per prompt)
bash agents-evals/run_eval.sh report --variance

# Near-miss accuracy
bash agents-evals/run_eval.sh report --near-miss

# Everything
bash agents-evals/run_eval.sh report --all
```

## Sweep configuration

`run_config.yaml` defines what to run:

```yaml
models:
  - id: deepseek-v4
    provider: opencode/deepseek-v4-flash-free
  - id: hy3-free
    provider: opencode/hy3-free

modes:
  - with-skills    # skills loaded normally
  - bare           # skills hidden for baseline comparison

runs: 3            # number of times to repeat each prompt
```

Edit this file to add/remove models, modes, or change the repetition count.

## How bare mode works

When `--mode bare` is used, `runner.py` does the following before each prompt:

1. Renames `AGENTS.md` → `AGENTS.md.bak` (removes skill guidance from context)
2. Renames `CLAUDE.md` → `CLAUDE.md.bak`
3. Moves `.opencode/skills/` → `.opencode/skills.bak/`
4. Moves `.agents/skills/` → `.agents/skills.bak/`
5. Moves `.claude/skills/` → `.claude/skills.bak/`

After the prompt run completes, everything is restored. This ensures the
model cannot load any PINA skills, giving a clean baseline for measuring
skill value-add.

## Runner CLI reference

```
python agents-evals/runner.py [options]

Options:
  --sweep CONFIG        Run full sweep from a YAML config file
  --prompt ID           Single prompt ID to run
  --model PROVIDER      Model provider string (e.g. opencode/deepseek-v4-flash-free)
  --model-id ID         Short label for filenames (default: last part of --model)
  --mode MODE           "with-skills" or "bare" (default: with-skills)
  --run N               Run number (default: 1)
  --runs N              Number of runs when using --model (default: 1)
  --check               Run checker after each evaluation
  --report              Run report after sweep completes
  --check-only          Run checker on all unchecked transcripts
  --timeout SECONDS     Per-prompt timeout (default: 120)
  --verbose             Print model responses live to stderr
```

## Metrics glossary

| Metric | What it measures |
|--------|------------------|
| **Skill precision** | Of skills loaded, how many were relevant |
| **Skill recall** | Of skills expected, how many were loaded |
| **Skill F1** | Harmonic mean of precision and recall |
| **Assertion pass rate** | Fraction of per-prompt assertions that passed |
| **Code compilability** | Fraction of code blocks that parse without syntax errors |
| **Code executability** | Fraction of code blocks that run without exceptions |
| **Near-miss accuracy** | Fraction of near-miss prompts correctly ignored |
| **Skill delta** | Δ(F1, pass rate, compile rate, exec rate) between with-skills and bare |
| **Token consumption** | Total tokens used per prompt (input + output + reasoning). Helps compare cost between modes. Bare mode tends to use **more** tokens because the model explores the codebase directly instead of following skill guidance. |
| **Wall time** | Clock time per prompt. Useful for comparing efficiency. |

## Near-miss prompts

`prompts/near-miss.yaml` contains prompts that should NOT trigger any PINA
skill. These test for over-eager triggering — e.g., mentioning "ODE" in a
general methods question shouldn't pull in `pina-workflow`.

## Prompt structure

```yaml
prompts:
  - id: causality-pinn
    text: "My PDE is time-dependent..."
    skills_expected: ["select-solver"]
    near_miss: false
    assertions:
      - id: suggests_causal_solver
        type: text_contains
        values: ["CausalPINN"]
        min_count: 1
      - id: code_compiles
        type: code_compiles
      - id: code_runs
        type: code_runs
        kwargs:
          max_epochs: 2
          accelerator: cpu
```

### Multi-turn prompts (handling follow-up questions)

PINA skills are conversational — they may ask follow-up questions. For fully
automated evaluation, each prompt can specify pre-baked answers via the
`turns` field:

```yaml
prompts:
  - id: navier-stokes-sensor
    text: "I have pressure/velocity sensor readings from a pipe..."
    turns:
      - "2D axisymmetric pipe, radius 0.1m, length 1m, Re=100."
      - "10 sensors evenly spaced along the axis."
      - "Default optimizer, 500 epochs."
    max_turns: 3  # optional, limits how many turns to use
```

The runner sends the main prompt, then pipes each turn via `opencode run --continue`.
All turns are combined into a single transcript with turn markers, and token
counts are accumulated across the entire conversation.

### Assertion types

| Type | What it checks |
|------|----------------|
| `text_contains` | Keyword/phrase anywhere in transcript |
| `code_regex` | Regex match inside extracted ` ```python ` blocks |
| `code_imports` | Specific Python imports present via AST |
| `code_compiles` | All code blocks have valid Python syntax |
| `code_runs` | All code blocks execute without exception (patched: `max_epochs=2`, `accelerator="cpu"`) |
| `not_present` | Specified keywords absent (used for near-miss) |

## Requirements

- Python 3.8+
- PyYAML (`pip install pyyaml`)
- opencode CLI (`curl -fsSL https://opencode.ai/install.sh | sh`)
