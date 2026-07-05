---
name: create-problem
description: >-
  Entry point for creating PINA problems. Routes to sub-skills based on problem
  type (data-driven vs physics-driven), problem class selection, domain setup,
  equations, conditions, and discretisation.
license: MIT
compatibility: opencode, codex, claude
metadata:
  audience: users
  workflow: problem-creation
---

# Create a PINA Problem — Entry Point

> [!IMPORTANT]
> Read [RULES.md](../RULES.md) before using this skill — it applies to all skills.

This is the **entry-point skill** for building PINA Problems. It selects the
problem type and routes to sub-skills for the deep work.

## Overview

A PINA problem is a Python class that inherits from one or more of:

| Base class            | When to use                                      |
|-----------------------|--------------------------------------------------|
| `BaseProblem`         | Data-driven (supervised / unsupervised) problems |
| `SpatialProblem`      | PDE/ODE depending only on spatial coordinates    |
| `TimeDependentProblem`| Problems with a time dimension                   |
| `ParametricProblem`   | Problems with parametric dependencies           |
| `InverseProblem`      | Problems with unknown physical parameters        |

Problems can **mix** base classes via multiple inheritance (e.g.,
`SpatialProblem` + `TimeDependentProblem` for space-time PDEs).

## Required attributes per problem type

| Base class(es)         | Must define                                       |
|------------------------|---------------------------------------------------|
| `BaseProblem`          | `input_variables`, `output_variables`, `conditions` with `input`/`target` |
| `SpatialProblem`       | `output_variables`, `spatial_domain`, `conditions` |
| `TimeDependentProblem` | `output_variables`, `temporal_domain`, `conditions` |
| `ParametricProblem`    | `output_variables`, `parameter_domain`, `conditions` |
| `InverseProblem`       | `output_variables`, `unknown_parameter_domain`, `conditions` |

## Interactive flow

### Step 1 — Problem nature

> Is your problem **data-driven** (you have input/target data) or
> **physics-driven** (you have a PDE/ODE with known equations)?

**Data-driven** → use `BaseProblem`. Load the **condition-setup** sub-skill for
data types and conditions.

**Physics-driven** → go to Step 2.

If the user is unsure:
- *Data-driven*: Pairs `(input, target)`, model learns to map one to the other.
- *Physics-driven*: Differential equation, model minimises residual at
  collocation points.

### Step 2 — Select domain type(s)

> Does your problem involve:
> - **Spatial variables only** (e.g. `x`, `y`, `z`)? → `SpatialProblem`
> - **Time** as well? → also inherit `TimeDependentProblem`
> - **Parameters** that vary? → also inherit `ParametricProblem`
> - **Unknown parameters** to be discovered? → also inherit `InverseProblem`

Choose the base class(es) that match.

### Step 3 — Delegate to sub-skills

1. Load **define-domains** to create the domain objects (`spatial_domain`,
   `temporal_domain`, etc.).
2. Load **define-equations** to define PDEs/ODEs and boundary conditions.
3. Load **condition-setup** to bind equations/conditions to domains.
4. Return to this skill after conditions are defined to handle **discretisation**
   (see Step 7 in define-domains) and final verification.

## Templates

### Template 1: Data-driven (supervised)

```python
import torch
from pina import Condition, LabelTensor
from pina.problem import BaseProblem

input_data = LabelTensor(torch.randn(100, 1), "x")
target_data = LabelTensor(torch.randn(100, 1), "y")

class MySupervisedProblem(BaseProblem):
    input_variables = ["x"]
    output_variables = ["y"]
    conditions = {
        "data": Condition(input=input_data, target=target_data),
    }

problem = MySupervisedProblem()
```

### Template 2: Purely spatial (Poisson-like)

```python
from pina.problem import SpatialProblem
from pina.domain import CartesianDomain
from pina import Condition
from pina.equation import Equation
from pina.equation.zoo import FixedValue

class MySpatialProblem(SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})

    domains = {
        "D": spatial_domain,
        "boundary": spatial_domain.partial(),
    }

    conditions = {
        "boundary": Condition(domain="boundary", equation=FixedValue(0.0)),
        "D": Condition(domain="D", equation=Equation(my_pde)),
    }

    def solution(self, pts):
        ...
```

### Template 3: Space-time (Burgers-like)

```python
from pina.problem import SpatialProblem, TimeDependentProblem
from pina.domain import CartesianDomain
from pina import Condition
from pina.equation import Equation
from pina.equation.zoo import FixedValue

class MySpaceTimeProblem(TimeDependentProblem, SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-1, 1]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    domains = {
        "D": spatial_domain.update(temporal_domain),
        "ic": spatial_domain.update(CartesianDomain({"t": 0})),
        "boundary": spatial_domain.partial().update(temporal_domain),
    }

    conditions = {
        "boundary": Condition(domain="boundary", equation=FixedValue(0.0)),
        "ic": Condition(domain="ic", equation=Equation(initial_cond)),
        "D": Condition(domain="D", equation=Equation(my_pde)),
    }

    def solution(self, pts):
        ...
```

### Template 4: Inverse problem

```python
from pina.problem import SpatialProblem, InverseProblem
from pina.domain import CartesianDomain
from pina import Condition
from pina.equation import Equation
from pina.equation.zoo import FixedValue

class MyInverseProblem(SpatialProblem, InverseProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-2, 2], "y": [-2, 2]})
    unknown_parameter_domain = CartesianDomain({"mu1": [-1, 1], "mu2": [-1, 1]})

    domains = {
        "D": spatial_domain,
        "boundary": spatial_domain.partial(),
    }

    conditions = {
        "boundary": Condition(domain="boundary", equation=FixedValue(0.0)),
        "D": Condition(domain="D", equation=Equation(laplace_equation)),
        "data": Condition(input=input_data, target=target_data),
    }
```

### Template 5: Parametric problem

```python
from pina.problem import SpatialProblem, ParametricProblem
from pina.domain import CartesianDomain

class MyParametricProblem(SpatialProblem, ParametricProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1]})
    parameter_domain = CartesianDomain({"mu": [0.5, 2.0]})

    domains = {
        "D": spatial_domain.update(parameter_domain),
        ...
    }
    ...
```

## Checklist

- [ ] `output_variables` is a `list[str]` naming the model outputs
- [ ] For data-driven: `input_variables` is a `list[str]` naming the inputs
- [ ] For physics-driven: all required domains are defined
      (`spatial_domain`, `temporal_domain`, `parameter_domain`, or
      `unknown_parameter_domain` as appropriate)
- [ ] `domains` dict contains a key for every domain referenced in conditions
- [ ] If equation was unknown: searched the web, presented the found
      formulation to the user, and confirmed before using
- [ ] PDE problem includes both the equation AND boundary/initial conditions
- [ ] `problem.discretise_domain(n=..., mode=..., domains=...)` called for
      each physics domain
- [ ] `problem.are_all_domains_discretised` is `True` after discretisation
- [ ] (Optional) `solution(pts)` method defined with correct analytical
      solution returning `LabelTensor` with `output_variables` labels
