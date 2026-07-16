---
name: define-equations
description: >-
  Define PDEs, ODEs, boundary conditions, and custom equations for PINA physics-driven
  problems. Covers the equation zoo, custom equation functions, and inverse problem
  equation signatures.
license: MIT
compatibility: opencode, codex, claude
metadata:
  audience: users
  workflow: problem-creation
---

# Define Equations for a PINA Problem

> [!IMPORTANT]
> Read [RULES.md](../RULES.md) before using this skill — it applies to all skills.
> This is a sub-skill of **create-problem**. Load the entry-point skill first.

Use this skill when the problem is **physics-driven** and you need to define
the governing equations and boundary conditions.

## Step 1 — Check the equation zoo

> What physical equations apply?

First check if the equation is available in the built-in zoo:

| Zoo class                     | Equation                         |
|-------------------------------|----------------------------------|
| `FixedValue(value)`           | `u - v = 0` (Dirichlet BC)       |
| `FixedGradient(value)`        | `∂u/∂n - v = 0` (Neumann BC)     |
| `FixedFlux(value)`            | `div(u) - v = 0` (flux BC)       |
| `FixedLaplacian(value)`       | `Δu - v = 0` (Laplacian BC)      |
| `PoissonEquation(forcing_f)`  | `Δu = f`                         |
| `BurgersEquation(nu)`         | `u_t + u u_x = ν u_xx`          |
| `AdvectionEquation(beta)`     | `u_t + β·∇u = 0`                |
| `AllenCahnEquation(ε, α)`     | `u_t = ε u_xx + α (u - u³)`     |
| `DiffusionReactionEquation(λ)`| `u_t = λ Δu + f`                |
| `HelmholtzEquation(k)`        | `Δu + k² u = f`                 |
| `AcousticWaveEquation(c)`     | `u_tt = c² Δu`                  |

If the equation matches one of these, use it directly.

## Step 2 — Unknown equation

If the equation is **not** in the zoo:

1. **Check** if the equation is present in `.opencode/skills/define-equations/famous_odes_pdes.json`,
   if not present **search the web** for the equation form, common PINN implementation, and known
   boundary/initial conditions.
2. Present the found formulation to the user and ask:
   > I found this equation: `<formulation>`. Should I use this or would you
   > like to provide your own?
3. If the user provides their own, use that instead.

## Step 3 — Define boundary conditions

> What boundary conditions apply? (Dirichlet, Neumann, Robin, periodic, etc.)

A well-posed PDE problem needs **both** the PDE and its boundary/initial
conditions. Common types:

- **Dirichlet**: `FixedValue(value)` — imposes `u = value` on the boundary
- **Neumann**: `FixedGradient(value)` — imposes `∂u/∂n = value`
- **Flux**: `FixedFlux(value)` — imposes `div(u) = value`
- **Laplacian**: `FixedLaplacian(value)` — imposes `Δu = value`

For zoo boundary conditions, use `components` and `d` parameters when the
problem has multiple output variables:

```python
FixedGradient(0.0, components=["theta"], d=["t"])
```

## Step 4 — Custom equation functions

Import the required utilities:

```python
from pina.operator import grad
from pina.equation import Equation
```

### Standard PDE (2 arguments)

```python
def my_pde(input_, output_):
    u_x = grad(output_, input_, components=["u"], d=["x"])
    u = output_.extract(["u"])
    return u_x - u
```

### Inverse problem (3 arguments)

When the problem is also an `InverseProblem`, the equation function receives a
third argument `params_` (a dict of unknown parameters):

```python
def my_pde_inverse(input_, output_, params_):
    f = torch.exp(-2 * (input_["x"] - params_["mu1"])**2)
    return laplacian(output_, input_, components=["u"], d=["x"]) - f
```

## Step 5 — Build conditions

Wrap equations in `Condition` objects. Return to the **condition-setup** or
**create-problem** skill for this step.

```python
from pina import Condition

conditions = {
    "interior": Condition(domain="D", equation=Equation(my_pde)),
    "dirichlet_bc": Condition(domain="boundary", equation=FixedValue(0.0)),
    "neumann_bc": Condition(domain="boundary", equation=FixedGradient(0.0)),
    "ic": Condition(domain="t0", equation=Equation(initial_cond)),
}
```

## Checklist

- [ ] Equation identified (zoo class or custom `Equation`)
- [ ] If equation was unknown: searched the web, presented the found
      formulation to the user, and confirmed before using
- [ ] Both the PDE and boundary/initial conditions are defined
- [ ] Custom equations use correct function signature:
      `(input_, output_)` for standard, `(input_, output_, params_)` for inverse
- [ ] Operator calls use `LabelTensor` with correct `components` and `d` parameter names
