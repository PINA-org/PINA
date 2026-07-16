---
name: select-model
description: >-
  Guides users through selecting, configuring, or creating a PyTorch model
  for a PINA problem. Use when the user asks about models, architecture,
  neural network choice, or how to build/choose a model for their PINA
  problem. Also triggers on phrases like "what model should I use",
  "choose a model", "create a model", "neural network", "architecture",
  "model selection", or when the user mentions specific model names
  (FeedForward, DeepONet, FNO, PirateNet, etc.).
license: MIT
compatibility: opencode, codex, claude
metadata:
  audience: users
  workflow: model-creation
---

# Select a Model for a PINA Problem

> [!IMPORTANT]
> Read [RULES.md](../RULES.md) before using this skill — it applies to all skills.

Use this skill to choose the right neural network architecture for a PINA
problem, configure its parameters, and optionally enhance it with embeddings
or blocks.

## Step 1 — Understand the problem together

Start by gathering enough context to make a reasoned recommendation. Ask the
user these questions conversationally — one at a time, adapting to their
answers — rather than firing them all at once.

- **Input/output**: What variables go in and come out? (e.g., input: x, y, t
  → output: u, v). If they have a PINA problem object, extract directly:
  `len(problem.input_variables)` / `len(problem.output_variables)`.
- **Data shape**: What does a single sample look like? Scalar values per
  point? A field on a grid? A graph? Sensor readings + coordinates?
- **Data structure** (determines what operations fit): A flat table of
  points (linear / MLP layers), a regular grid (conv1d/conv2d/conv3d, or
  FNO), an irregular mesh or point cloud (message passing / graph nets,
  or coordinate-based MLPs like PINNs), or sequences / time series (RNN /
  transformer)?
- **Problem type**: Are you solving a PDE (physics-informed), learning an
  operator (map function to function), fitting a curve/surface, or
  discovering equations from data?
- **Constraints**: Roughly how much data? Any compute or time limits? Is
  accuracy-critical or is a rough baseline enough?

If the user cannot describe their problem at all, reply:

> I need at least the input/output dimensions to suggest an architecture.
> What are the input variables and what are the outputs you want to predict?

## Step 2 — Reason about architecture

Take the context from Step 1 and reason about which architecture family fits
best. Present the reasoning to the user and invite their input rather than
simply prescribing.

### Available building blocks

Models live in PINA's model (`pina.model`), with smaller blocks in
`pina.model.block`. Browsing these modules is the best way to discover what's
available — the sections below discuss how to choose. 

### Reasoning guide

Work through these considerations in dialogue with the user:

1. **Operator learning?** — mapping between function spaces? That points to
   FNO (regular grid), DeepONet/MIONet (sensor→field), or neural operators
   (Averaging/LowRank/Graph). Ask about the data geometry.

2. **Message-passing / GNN?** — input is a graph with explicit connectivity
   (mesh, point cloud, network)?
   - PINA has many GNN based blocks in `pina.model.block.message_passing`.

3. **Equation discovery?** — SINDy or similar. Make sure the user knows this
   discovers symbolic expressions, not just a black-box surrogate.

4. **Spline fitting?** — Spline for 1D, SplineSurface for 2D, or
   VectorizedSpline for independent per-feature splines. Typically explicit
   fitting, not deep learning.

5. **Standard pointwise mapping?** — This is where most PDE problems land.
   Within this family, consider:
   - **Complexity**: Start simple (FeedForward). Only escalate if it
     struggles — PirateNet for multi-scale, ResidualFeedForward for depth.
   - **Data regime**: Lots of collocation points → deeper/larger MLP works.
     Very little data → KAN can be parameter-efficient.
   - **Multi-scale / high-frequency**: FourierFeatureEmbedding (Step 3) is
     a cheap add-on to any MLP. PirateNet bundles it natively.

6. **Custom `nn.Module`** — When none of the above fit, use pure PyTorch.
   This is common for:
   - Multi-input/multi-output architectures PINA doesn't have
   - Pretrained encoders (CNNs, transformers)
   - Novel research architectures
   - Lightweight models where PINA's wrapping adds unnecessary overhead

   The only requirement: it must be a valid `torch.nn.Module` that accepts
   a tensor/graph and returns a tensor/graph. With PINA solvers,
   keep the constructor parameter names `input_dimensions` and 
   `output_dimensions` so labels are handled automatically
   (or use `nn.Module` directly without PINA).

## Step 3 — Enhancements and embeddings

These are add-ons to improve convergence or enforce structure. Discuss
whether the user's problem benefits from them.

### Fourier Feature Embedding (multi-scale)

Use when the PDE solution has features at multiple length scales, or when
a plain MLP converges slowly (spectral bias).

```python
from pina.model.block import FourierFeatureEmbedding

embedding = FourierFeatureEmbedding(
    input_dimension=input_dim,
    output_dimension=128,  # must be even
    sigma=2.0,
)
```

Wrap any model manually:

```python
class WithEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, inner_size=64):
        super().__init__()
        self.embedding = FourierFeatureEmbedding(input_dim, 128)
        self.net = nn.Sequential(
            nn.Linear(128, inner_size), nn.Tanh(),
            nn.Linear(inner_size, output_dim),
        )

    def forward(self, x):
        return self.net(self.embedding(x))
```

### Periodic Boundary Embedding

Hard-code periodicity so the model never needs to learn it:

```python
from pina.model.block import PeriodicBoundaryEmbedding

embedding = PeriodicBoundaryEmbedding(
    input_dimension=2,
    periods={"x": 2 * 3.14159},
)
```

### Residual / Orthogonal blocks

- `ResidualBlock(input_dim, output_dim, hidden_dim, activation)` — stacks
  layers with a skip connection. Use inside a custom model.
- `OrthogonalBlock(dim=-1)` — orthonormalize features along a dimension.
  Helps training stability.

Both are in `pina.model.block`.

### Message-passing blocks (graph models)

For graph-structured inputs, PINA provides blocks that follow the same
pattern: update node features by aggregating neighbor information. Available
in `pina.model.block` (top-level) and `pina.model.block.message_passing`:

| Block | Source | Description |
|---|---|---|
| `InteractionNetworkBlock` | `.message_passing` | Standard encode–process–decode interaction network |
| `DeepTensorNetworkBlock` | `.message_passing` | Higher-order tensor message passing |
| `EnEquivariantNetworkBlock` | `.message_passing` | Energy-conserving equivariant message passing |
| `EquivariantGraphNeuralOperatorBlock` | `.message_passing` | SE(3)-equivariant neural operator block |
| `RadialFieldNetworkBlock` | `.message_passing` | Radial-basis field convolution on point clouds |
| `GNOBlock` | `.block` (top-level) | Graph neural operator kernel integration |

Prefer the built-in `GraphNeuralOperator` (in `pina.model`) if it matches
your problem — it composes lifting, message passing, and projection into
one class. Drop down to these blocks when you need a custom arrangement.

## Step 4 — Build and test

### Construction pattern

Prefer PINA's built-in models when they match (less boilerplate). Use
pure `nn.Module` when they don't.

```python
# PINA model
from pina.model import FeedForward
model = FeedForward(input_dimensions=3, output_dimensions=3, inner_size=64, n_layers=4, func=nn.Tanh)

# Pure torch custom model
class MyModel(nn.Module):
    def __init__(self, input_dimensions, output_dimensions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dimensions, 128), nn.Tanh(),
            nn.Linear(128, output_dimensions),
        )
    def forward(self, x):
        return self.net(x)

model = MyModel(input_dimensions=3, output_dimensions=3)

# Graph model: lifting → message passing → projection
from pina.model.block import GNOBlock, FeedForward

class GraphModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lifting = FeedForward(input_dim, hidden_dim)
        self.gno = GNOBlock(...)
        self.projection = FeedForward(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.lifting(x)
        x = self.gno(x, edge_index, edge_attr)
        return self.projection(x)
```

### Validation

- Forward pass — tensor model: `model(torch.randn(batch, input_dim))` →
  `(batch, output_dim)`.
- Forward pass — graph model: `model(x, edge_index, edge_attr)` with dummy
  tensors of the right shapes, or a `torch_geometric.data.Data` object if
  the model expects one. The output should match the expected node-feature
  shape.
- If using PINA solvers: confirm the solver can call `model.forward()` with
  the actual condition tensors (same dtype, device, shapes).
- If using custom `nn.Module` with PINA: keep `input_dimensions` and
  `output_dimensions` as constructor kwargs for automatic label extraction.

## Checklist

- [ ] Input/output dimensions determined (from problem object or user Q&A)
- [ ] Architecture reasoning discussed with the user, not prescribed
- [ ] Model class selected (PINA built-in, or custom `nn.Module`)
- [ ] Embeddings/enhancements added if the problem needs them (multi-scale,
      periodic boundaries, residual connections)
- [ ] Model passes a forward and backward pass smoke test with **correct** shapes