---
name: condition-setup
description: >-
  Set up conditions for PINA problems. Covers data types (LabelTensor, Graph,
  PyG Data), time series conditions, binding equations to domains, and
  data-driven input→target mapping.
license: MIT
compatibility: opencode, codex, claude
metadata:
  audience: users
  workflow: problem-creation
---

# Set Up Conditions for a PINA Problem

> [!IMPORTANT]
> Read [RULES.md](../RULES.md) before using this skill — it applies to all skills.
> This is a sub-skill of **create-problem**. Load the entry-point skill first.

Use this skill to define `Condition` objects that bind data, equations, or
time-series windows to the problem.

## Step 1 — Determine the condition type

Three kinds of conditions exist in PINA:

| Kind                        | When to use                                      |
|-----------------------------|--------------------------------------------------|
| **Physics-on-domain**       | PDE/ODE residual on a sampled domain             |
| **Data-driven**             | Input→target mapping (supervised)                |
| **Time series**             | Rolling-window forecasting                       |

## Step 2 — Data types (data-driven only)

If the problem is data-driven, ask:

> What data type are you using?

Available data types for `Condition(input=..., target=...)`:
- **`LabelTensor` / `torch.Tensor`** — standard tensor data (most common)
- **`Graph`** — PINA's built-in graph structure (from `pina import Graph`)
- **`Data`** — PyTorch Geometric `Data` object (from `torch_geometric.data import Data`)

All three types are accepted directly as `input`/`target`.

## Step 3 — Build conditions

### Physics-on-domain

Conditions map domain names (sampled later via `discretise_domain`) or explicit
point tensors to equations:

```python
from pina import Condition

# Option 1: reference a domain by name (sampled later)
conditions = {
    "boundary": Condition(domain="boundary", equation=FixedValue(0.0)),
    "interior": Condition(domain="D", equation=Equation(my_pde)),
}

# Option 2: provide explicit points
conditions = {
    "data_pde": Condition(input=points_tensor, equation=Equation(my_pde)),
}
```

### Data-driven (supervised)

```python
conditions = {
    "data": Condition(input=input_tensor, target=target_tensor),
}
```

### Time series forecasting

If the user has time series data, ask whether they want **standard supervised**
or **time-series** conditions:

```python
from pina import Condition

# Standard supervised
Condition(input=ts_tensor, target=target_tensor)

# Time series (input is 3D: [batch, n_windows, features])
Condition(
    input=ts_tensor,
    n_windows=10,
    unroll_length=5,
    randomize=True,
)

# Graph time series
Condition(
    input=graph_ts_data,
    n_windows=10,
    unroll_length=5,
    key="some_key",
)
```

Parameters:
- `n_windows` — number of rolling windows
- `unroll_length` — prediction horizon per window
- `randomize` — shuffle window order
- `key` — key for graph time series data

## Step 4 — Integrate with the problem class

Conditions become a class-level dict on the problem:

```python
class MyProblem(SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1]})

    domains = {
        "D": spatial_domain,
        "boundary": spatial_domain.partial(),
    }

    conditions = {
        "boundary": Condition(domain="boundary", equation=FixedValue(0.0)),
        "D": Condition(domain="D", equation=Equation(my_pde)),
    }
```

## Checklist

- [ ] For data-driven: confirmed data type (`LabelTensor`, `torch.Tensor`,
      `Graph`, or PyG `Data`)
- [ ] For data-driven: `input_variables` is a `list[str]` naming the inputs
- [ ] For time series: `n_windows`, `unroll_length`, and optional `key`
      are set correctly
- [ ] Each `Condition` uses valid keyword arguments:
  - `Condition(domain=..., equation=...)` for physics-on-domain
  - `Condition(input=..., equation=...)` for physics-on-points
  - `Condition(input=..., target=...)` for data-driven
  - `Condition(input=..., n_windows=..., unroll_length=...)` for time series
- [ ] `domains` dict has an entry for every domain name used in conditions
