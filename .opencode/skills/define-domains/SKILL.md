---
name: define-domains
description: >-
  Create and manipulate domains for PINA physics-driven problems. Covers domain
  types (Cartesian, Ellipsoid, Simplex), set operations, partial/update methods,
  and domain discretisation.
license: MIT
compatibility: opencode, codex, claude
metadata:
  audience: users
  workflow: problem-creation
---

# Define Domains for a PINA Problem

> [!IMPORTANT]
> Read [RULES.md](../RULES.md) before using this skill — it applies to all skills.
> This is a sub-skill of **create-problem**. Load the entry-point skill first.

Use this skill to create spatial, temporal, and parameter domains, and to
discretise them for training.

## Step 1 — Create domains

For each domain type, ask:

> What are the variable names and their ranges?

If the user does not specify a domain, ask for:
1. Variable name (e.g. `x`)
2. Lower bound (e.g. `0`)
3. Upper bound (e.g. `1`)

### Domain types available

```python
from pina.domain import CartesianDomain, EllipsoidDomain, SimplexDomain
```

| Domain type        | Description                          | Example                                   |
|--------------------|--------------------------------------|-------------------------------------------|
| `CartesianDomain`  | Hyperrectangle (most common)         | `CartesianDomain({"x": [0, 1], "y": [0, 1]})` |
| `EllipsoidDomain`  | Hyperellipsoid                       | `EllipsoidDomain({"x": [0, 1], "y": [0, 1]})` |
| `SimplexDomain`    | Simplex defined by vertices          | `SimplexDomain(vertices=[...])`           |

`CartesianDomain` supports sampling modes: `random`, `grid`, `chebyshev`,
`latin`/`lh`.

### Set operations on domains

```python
from pina.domain import Union, Intersection, Difference, Exclusion

combined = Union(domain_a, domain_b)
overlap = Intersection(domain_a, domain_b)
subtracted = Difference(domain_a, domain_b)
excluded = Exclusion(domain_a, domain_b)
```

## Step 2 — Domain methods for problem setup

### `partial()` — extract boundary

Creates a sub-domain representing the boundary of the parent domain:

```python
spatial_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})
boundary = spatial_domain.partial()  # boundary of the square
```

### `update()` — combine domains (space + time)

Creates the Cartesian product of two domains. Essential for space-time problems:

```python
spatial_domain = CartesianDomain({"x": [-1, 1]})
temporal_domain = CartesianDomain({"t": [0, 1]})

interior = spatial_domain.update(temporal_domain)
# Equivalent to CartesianDomain({"x": [-1, 1], "t": [0, 1]})
```

Common patterns:

```python
domains = {
    "D": spatial_domain.update(temporal_domain),   # space-time interior
    "ic": spatial_domain.update(CartesianDomain({"t": 0})),  # initial condition
    "boundary": spatial_domain.partial().update(temporal_domain),  # moving boundary
}
```

## Step 3 — Discretise domains (sampling)

After the problem class is fully defined, sample points from each domain:

```python
problem.discretise_domain(n=5000, mode="random", domains=["D"])
problem.discretise_domain(n=500, mode="random", domains=["boundary"])
```

| Mode          | Description                           |
|---------------|---------------------------------------|
| `"random"`    | Uniform random sampling (default)     |
| `"latin"`/`"lh"` | Latin hypercube sampling          |
| `"grid"`      | Regular grid points                   |
| `"chebyshev"` | Chebyshev nodes (good for polynomials)|

After all domains are discretised:

```python
problem.move_discretisation_into_conditions()
assert problem.are_all_domains_discretised
```

## Checklist

- [ ] All required domains are defined (`spatial_domain`, `temporal_domain`,
      `parameter_domain`, or `unknown_parameter_domain` as appropriate)
- [ ] `domains` dict contains a key for every domain referenced in conditions
- [ ] For space-time problems: `spatial_domain.update(temporal_domain)` used
      to build the interior domain
- [ ] For boundary conditions: `spatial_domain.partial()` used correctly
- [ ] `problem.discretise_domain(n=..., mode=..., domains=...)` called for
      each physics domain
- [ ] `problem.move_discretisation_into_conditions()` called before training
- [ ] `problem.are_all_domains_discretised` is `True` after discretisation
