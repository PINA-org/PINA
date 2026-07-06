# PINA — Physics-Informed Neural Architectures

A PyTorch library for solving differential equations with neural networks (PINNs).

## Quick Reference

### Workflow: Problem → Model → Solver → Trainer

```python
problem = MyProblem()
problem.discretise_domain(n=256, mode="grid")
model = FeedForward(input_dimensions=2, output_dimensions=1)
solver = PINN(problem=problem, model=model)
trainer = Trainer(solver=solver, max_epochs=1000)
trainer.train()
```

### Problem types
`SpatialProblem`, `TimeDependentProblem`, `ParametricProblem`, `InverseProblem`.
Define `output_variables`, `spatial_domain`/`temporal_domain`, and `conditions` (dict of `Condition`).

### Condition types
- `Condition(domain=..., equation=...)` — physics residual on sampled domain
- `Condition(input=..., equation=...)` — physics residual at fixed points
- `Condition(input=..., target=...)` — supervised data
- `Condition(input=..., n_windows=..., unroll_length=...)` — time series

### Domains
`CartesianDomain`, `EllipsoidDomain`, `SimplexDomain`.
Set ops: `Union`, `Intersection`, `Difference`, `Exclusion`.
Discretise with `problem.discretise_domain(n, mode)` where mode is `"grid"`, `"random"`, `"lh"`, `"chebyshev"`.

### Solvers
`PINN`, `CausalPINN`, `SelfAdaptivePINN`, `CompetitivePINN`, `GradientPINN`, `RBAPINN`,
`SupervisedSolver`, `AutoregressiveSolver`. Ensembles via `*EnsembleSolver` variants.

### Models
`FeedForward`, `ResidualFeedForward`, `PirateNet`, `DeepONet`, `MIONet`, `FNO`,
`KolmogorovArnoldNetwork`, `GraphNeuralOperator`, `SINDy`, and more.

### Equation zoo
`PoissonEquation`, `HelmholtzEquation`, `BurgersEquation`, `AdvectionEquation`,
`AllenCahnEquation`, `AcousticWaveEquation`, `DiffusionReactionEquation`.
Boundary: `FixedValue`, `FixedGradient`, `FixedFlux`, `FixedLaplacian`.

### Key utilities
- `LabelTensor` — `torch.Tensor` with named columns; index via `.extract(["x", "y"])`
- Differential operators: `grad`, `div`, `laplacian`, `advection` — not cached, compute once
- `Trainer` wraps `lightning.pytorch.Trainer`; handles DataModule, batching, device placement

## Skills

Skills under `.opencode/skills/<name>/SKILL.md` give step-by-step guidance.
Load the relevant skill when the user's request matches:

| Skill | When to load |
|-------|-------------|
| `pina-workflow` | User wants to solve a PDE/ODE or model a physical system — orchestrates the full session |
| `create-problem` | Defining a problem, setting up variables and outputs |
| `define-domains` | Creating and discretising domains |
| `define-equations` | Writing PDEs/ODEs or using the equation zoo |
| `condition-setup` | Binding equations to domains, data-driven conditions |
| `select-model` | Choosing or configuring a neural architecture |
| `select-solver` | Selecting the right solver for the problem |
| `select-trainer` | Configuring training parameters and callbacks |
| `create-skill` | User wants to create or edit a skill in this repo |
| `skill-sync-checker` | Auditing skill references against the codebase |

## Commands

```bash
pytest tests/ -x       # run tests
ruff check .           # lint
```

## Rules

- Never modify source code (`pina/`, `tests/`, `docs/`). Skills are read-only guidance.
- Stay conversational — walk through the workflow naturally.
