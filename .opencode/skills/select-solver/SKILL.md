---
name: select-solver
description: >-
  Guides users through selecting the right PINA solver for their problem, or
  creating a custom solver when no built-in fits. Use when the user asks "what
  solver should I use", "how do I train this", "which solver", "choose a
  solver", "pick a solver", "solver selection", "how do I set up training",
  "what's the right solver", or similar. Also triggers when the user mentions
  specific solver names (PINN, CausalPINN, SelfAdaptivePINN, SupervisedSolver,
  etc.) or asks about custom training loops / custom solvers.
license: MIT
compatibility: opencode, codex, claude
metadata:
  audience: users
  workflow: solver-selection
---

# Select a Solver for a PINA Problem

> [!IMPORTANT]
> Read [RULES.md](../RULES.md) before using this skill — it applies to all skills.

Use this skill to pick the correct PINA solver — or design a custom one — based
on your problem type, conditions, and training requirements.

PINA solvers are driven by **conditions** (data format) and **model count**
(single vs ensemble). The condition types in your `problem.conditions` dictate
which solver families are compatible; additional requirements (causality,
gradient enhancement, adaptive weighting) narrow the choice further.

## Step 1 — Understand the conditions

Start by identifying what kind of data your problem works with. Ask
conversationally if the user hasn't already built their problem object:

- **Time series?** — Your conditions use `n_windows` and `unroll_length`.
  The data has a sequential structure and the model rolls out predictions
  step by step.
- **Input → Target?** — You have input points and known target values.
  This is standard supervised / regression data: `Condition(input=..., target=...)`.
- **PDE equations + domains?** — Your conditions reference equations and
  domains/spatial points: `Condition(domain=..., equation=...)` or
  `Condition(input=..., equation=...)`.
- **Mixed?** — Many physics-informed problems mix equation conditions (for
  the PDE interior) with target conditions (for boundary data or
  observations).

You can inspect an existing problem object directly:

```python
for name, cond in problem.conditions.items():
    print(name, type(cond).__name__)
```

If the user hasn't defined conditions yet, ask if you can help setting up the
problem.

## Step 2 — Let the conditions drive the choice

Once the condition types are clear, map them to solver families. This is a
**decision tree**, not a menu — work through it with the user.

### Time series conditions

If every condition is (or includes) a `TimeSeriesCondition`:

```
→ AutoregressiveSingleModelSolver  (one model)
→ AutoregressiveEnsembleSolver     (multiple models, averaged)
```

No other solver family handles `TimeSeriesCondition`.

Key parameters:
- `eps` — noise injected during unrolling (regularisation)
- `unroll_length` — number of steps per forward pass

### Input → Target conditions only

If all conditions are `InputTargetCondition` (no equations, no domains):

```
→ SupervisedSingleModelSolver  (one model)
→ SupervisedEnsembleSolver     (N models with deep ensemble)
```

The `Ensemble` variant trains N independent copies and averages their
predictions at inference.

### Equation conditions (physics-informed)

If conditions include `InputEquationCondition` or `DomainEquationCondition`,
you're in the physics-informed family. Now ask two follow-up questions:

#### 1. How many models: one or multiple?

| Count | Options |
|-------|---------|
| Single model | `PhysicsInformedSingleModelSolver` (base) + its specialisations below |
| Multiple models (ensemble) | `PhysicsInformedEnsembleSolver` |

Multiple models trains N independent copies and averages predictions.

#### 2. Any special training requirement?

For the **single-model** physics-informed case, ask the user about these.
Each maps to a distinct solver:

| Requirement | Solver | What it does |
|---|---|---|
| Standard (no special needs) | `PhysicsInformedSingleModelSolver` | Plain PINN training |
| Gradient-enhanced | `GradientPhysicsInformedSingleModelSolver` | Adds gradient norm penalty — regularises the solution's derivatives. **Requires** `SpatialProblem` (needs `.spatial_variables`). |
| Causality in time | `CausalPhysicsInformedSingleModelSolver` | Applies causal temporal weighting so the solver learns forward in time. **Requires** `TimeDependentProblem`. |
| Per-point residual attention | `RBAPhysicsInformedSingleModelSolver` | Re-weights collocation points by residual magnitude across epochs (focus on hard regions). |
| Per-parameter adaptive weights | `SelfAdaptivePhysicsInformedSolver` | Learns a per-point weight through a second model (min-max optimisation). |
| Adversarial discriminator | `CompetitivePhysicsInformedSolver` | A discriminator model bets on point residuals; solver must fool it — minimax game. |

**Self-adaptive and competitive are not single-model solvers** — they manage
multiple optimisers internally. `SelfAdaptive` trains a weight network
alongside the model; `Competitive` trains a discriminator.

**Ensemble** + special requirements? The `PhysicsInformedEnsembleSolver`
does not layer gradient/causal/self-adaptive on top. For those combinations,
create a custom solver (Step 4).

## Step 3 — Assemble the solver

Once the solver is chosen, the construction pattern is consistent:

```python
from pina.solver import PhysicsInformedSingleModelSolver
# or any other solver

solver = PhysicsInformedSingleModelSolver(
    problem=problem,
    model=model,
    learning_rate=0.001,
    # optional:
    # loss=torch.nn.MSELoss(),
    # weighting=my_weighting,
    # scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(...),
    # batch_size=32,
)
```

## Step 4 — Creating a custom solver

When no built-in solver matches your requirements, PINA's architecture makes
it straightforward to compose one. The key building blocks are **mixins** —
reusable `training_step` and `_compute_condition_loss` overrides.

### When to create a custom solver

- You need a combination PINA doesn't provide (e.g., causal weighting +
  ensemble, or gradient-enhanced + competitive)
- Your loss computation has custom per-condition logic not covered by the
  existing `_compute_condition_loss` overrides
- You need a multi-model setup that isn't just an ensemble (e.g., two models
  exchanging information in a non-adversarial way)
- You need to override `training_step` for a non-standard optimiser loop

### How PINA solvers are composed

Every solver is a class hierarchy:

```
BaseSolver
 ├── training/val/test_step
 ├── _compute_condition_loss
 ├── _prepare_condition_data
 ├── _regularize_condition_loss
 └── _loss_from_residual

SingleModelSolver(BaseSolver)
 └── automatic_optimization=True, single forward

MultiModelSolver(BaseSolver)
 └── automatic_optimization=False, multiple optimisers

EnsembleSolver(BaseSolver)          — uses MultiModelSolver internally
 └── automatic_optimization=False, averages N models
```

Mixins override specific methods:

| Mixin | Overrides | Purpose |
|---|---|---|
| `PhysicsInformedMixin` | `validation_step`, `test_step` | Enables autograd in no-grad contexts |
| `GradientEnhancedMixin` | `_prepare_condition_data`, `_regularize_condition_loss` | Gradient penalty regularisation |
| `ResidualBasedAttentionMixin` | `_regularize_condition_loss` | Per-point attention weights |
| `AutoregressiveMixin` | `_loss_from_residual` | Step-wise adaptive loss |
| `ManualOptimizationMixin` | `training_step` | Disables automatic optimisation |
| `ConditionAggregatorMixin` | `training_step` | Iterates conditions, aggregates losses |

### Custom solver recipe

1. **Pick a base**: `SingleModelSolver` (auto-optim) or `MultiModelSolver`
   (manual optim for multiple optimisers) or `EnsembleSolver` (N-model avg).

2. **Mix in behaviours** by inheriting the mixins in the right order (mixins
   first so their method overrides take priority):

   ```python
   from pina.solver import SingleModelSolver
   from pina.solver.mixin import GradientEnhancedMixin, PhysicsInformedMixin

   class MyCustomSolver(
       PhysicsInformedMixin,      # 1st — ensures autograd in val/test
       GradientEnhancedMixin,     # 2nd — adds gradient penalty
       SingleModelSolver,         # 3rd — base training loop
   ):
       def __init__(self, problem, model, **kwargs):
           super().__init__(problem=problem, model=model, **kwargs)
   ```

3. **Override `_compute_condition_loss`** if you need custom per-condition
   logic (e.g., different loss functions per condition, special weighting).
   The signature is:

   ```python
   def _compute_condition_loss(self, condition, data, batch_idx):
       # data is a dict like {"input": tensor, "target": tensor}
       # condition is the Condition object
       ...
       return scalar_tensor
   ```

4. **Override `training_step`** only for fundamental changes to the optimiser
   loop (e.g., alternating min-max, multi-stage schedules):

   ```python
   def training_step(self, batch, batch_idx):
       # Custom loop: zero_grad → compute loss → manual_backward → step
       ...
   ```

### Example: Causal ensemble

```python
from pina.solver import EnsembleSolver

class CausalEnsemblePINN(EnsembleSolver):
    """Ensemble of models with causal temporal weighting."""
    def _compute_condition_loss(self, condition, data, batch_idx):
        # Inject causal weighting logic here
        # (iterate over time segments, apply causal mask)
        ...
```

When guiding a user through creating a custom solver, discuss their specific
need, identify which existing mixin or base class covers most of it, and then
describe only the method they need to override. Avoid generating the full
solver class unless the user explicitly asks for it.

## Templates

### Choosing from conditions

Use this pattern when the problem object already exists:

```python
from pina.solver import (
    SupervisedSingleModelSolver,
    PhysicsInformedSingleModelSolver,
    AutoregressiveSingleModelSolver,
    PhysicsInformedEnsembleSolver,
)

condition_types = {type(c).__name__ for _, c in problem.conditions.items()}

if "TimeSeriesCondition" in condition_types:
    solver = AutoregressiveSingleModelSolver(problem=problem, model=model)
elif "InputTargetCondition" in condition_types and not \
     ("InputEquationCondition" in condition_types or
      "DomainEquationCondition" in condition_types):
    solver = SupervisedSingleModelSolver(problem=problem, model=model)
else:
    solver = PhysicsInformedSingleModelSolver(problem=problem, model=model)
```

### Custom solver stub

```python
from pina.solver import SingleModelSolver

class MySolver(SingleModelSolver):
    """Custom solver for specialised loss logic."""

    def _compute_condition_loss(self, condition, data, batch_idx):
        # Your custom loss computation
        # data keys: "input", "target" (if InputTargetCondition)
        #            "input", "equation" (if InputEquationCondition)
        ...
        return loss

    def training_step(self, batch, batch_idx):
        # Only if you need a custom optimiser loop
        ...
```

## Checklist

- [ ] Condition types identified (time series / supervised / physics-informed / mixed)
- [ ] Condition-to-solver-family mapping discussed with the user
- [ ] Single-model vs ensemble decision made
- [ ] Special training requirements assessed (causality, gradient, adaptive, competitive)
- [ ] If custom solver: mixin composition pattern understood, only necessary overrides discussed
- [ ] Solver constructed with correct signature (problem, model/models, extra params)
