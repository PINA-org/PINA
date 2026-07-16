---
name: select-trainer
description: >-
  Guides users through configuring the PINA Trainer for solver training,
  including batching strategy, data splitting, and Lightning options. Use when
  the user asks "how do I train", "how to train", "set up training", "trainer",
  "Trainer", "train my model", "fit", "training loop", or similar. Also
  triggers when the user mentions batching, training configuration,
  train/val/test split, or is confused about how to start training.
license: MIT
compatibility: opencode, codex, claude
metadata:
  audience: users
  workflow: training-setup
---

# Configure Training for a PINA Solver

> [!IMPORTANT]
> Read [RULES.md](../RULES.md) before using this skill — it applies to all skills.

Use this skill to set up and run training once a solver has been chosen.

PINA's `Trainer` wraps Lightning's `Trainer` with PINA-specific defaults:
data splitting across conditions, batching strategies, device placement for
inverse parameters, and gradient tracking for physics-informed solvers.

## Step 1 — Understand what the trainer needs

The trainer needs three things. If any are missing, ask the user:

1. **Solver** (required) — already chosen and instantiated.
2. **Domain discretisation** — all `DomainEquationCondition` domains must be
   sampled before the trainer can create dataloaders.
3. **Training config** — epochs, batch size, accelerator, device.

### Domain discretisation

Physics-informed problems use `domain=...` in their conditions. The trainer
requires every such domain to have been sampled:

```python
problem.discretise_domain(n=1000, mode="random")
```

If you forget, the trainer raises a clear error listing which domains are
missing.

## Step 2 — Configure the trainer

The constructor has two groups of parameters: PINA-specific and Lightning
`**kwargs` passed through to the parent class.

### PINA-specific parameters

| Parameter | Default | What it controls |
|---|---|---|
| `solver` | — (required) | The solver instance to train |
| `batch_size` | `None` | `None` = full batch; int = mini-batches |
| `train_size` / `val_size` / `test_size` | `1.0 / 0.0 / 0.0` | Fraction split per condition |
| `batching_mode` | `"common_batch_size"` | How batches are built across conditions (see below) |
| `automatic_batching` | `False` | `True` = Lightning's default collation; `False` = direct subset retrieval |
| `num_workers` | `0` | Dataloader workers |
| `pin_memory` | `False` | Pin memory for faster GPU transfer |
| `shuffle` | `True` | Shuffle before splitting |

### Batching modes

The mode controls how condition data is assembled into batches:

| Mode | Behaviour | When to use |
|---|---|---|
| `"common_batch_size"` | Each condition supplies `batch_size` points per batch | Default — works for most cases |
| `"proportional"` | Batch sizes are scaled by condition dataset sizes | Unbalanced datasets (e.g., many interior points but few boundary points) |
| `"separate_conditions"` | Iterates through each condition separately | Each condition's data is heterogeneous (e.g., one is pointwise, another is a graph) |

### Common Lightning kwargs

These are passed as `**kwargs` and fully documented by PyTorch Lightning.
Key ones for PINA users:

| Kwarg | What it does |
|---|---|
| `max_epochs` | Number of training epochs |
| `accelerator` | `"cpu"`, `"gpu"`, `"mps"` (Apple Silicon) |
| `devices` | Device index or count (e.g., `1`, `[0]`, `"auto"`) |
| `precision` | `"16-mixed"`, `"32"`, `"64"`, `"bf16-mixed"` |
| `enable_progress_bar` | `True` / `False` |
| `gradient_clip_val` | Gradient clipping threshold |
| `callbacks` | List of Lightning callbacks (e.g., `ModelCheckpoint`, `EarlyStopping`) |

## Step 3 — Train and test

Usage is uniform regardless of solver type:

```python
from pina import Trainer

trainer = Trainer(
    solver=solver,
    max_epochs=1000,
    batch_size=32,
    accelerator="cpu",
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
)

trainer.train()   # Lightning fit()
trainer.test()    # Lightning test() — optional
```

## Templates

### Full-batch physics-informed

```python
problem.discretise_domain(n=5000, mode="random")

trainer = Trainer(
    solver=solver,
    max_epochs=10000,
    batch_size=None,          # full batch — no mini-batching
    accelerator="cpu",
)
trainer.train()
```

### Mini-batch supervised

```python
trainer = Trainer(
    solver=solver,
    max_epochs=500,
    batch_size=64,
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    accelerator="gpu",
    devices=1,
    num_workers=4,
    shuffle=True,
)
trainer.train()
trainer.test()
```

### Unbalanced conditions (proportional batching)

```python
trainer = Trainer(
    solver=solver,
    batch_size=512,
    batching_mode="proportional",
    max_epochs=1000,
)
```

## Checklist

- [ ] Solver already chosen and instantiated (see **select-solver** skill)
- [ ] All domains discretised: `problem.discretise_domain(n=..., mode="...")`
- [ ] Batch size decided: `None` for full-batch, an int for mini-batches
- [ ] Batching mode selected based on condition balance
- [ ] Train/val/test split configured
- [ ] Accelerator and device chosen
- [ ] Training started: `trainer.train()`
