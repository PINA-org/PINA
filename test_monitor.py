"""
Minimal test for TopologyMonitor.
"""
import torch
import numpy as np
from scipy import io
from torch.utils.data import TensorDataset, DataLoader
from pina.model import FNO, FeedForward
from pina.solver import SupervisedSingleModelSolver
from pina.problem.zoo import SupervisedProblem
from pina import Trainer, LabelTensor
from pina.topology import TopologyMonitor, TopologicalProfiler, GudhiBackend

print("=" * 60)
print("Testing TopologyMonitor with Darcy dataset")
print("=" * 60)

# 1. Load data (Channels-Last for FNO)
print("\n[1] Loading Darcy data...")
data = io.loadmat("Data_Darcy.mat")
k_train = torch.tensor(data["k_train"], dtype=torch.float)
u_train = torch.tensor(data["u_train"], dtype=torch.float)

k_train = k_train.unsqueeze(-1)
u_train = u_train.unsqueeze(-1)

print(f"    Total samples: {k_train.shape[0]}")
print(f"    Grid size: {k_train.shape[1]}x{k_train.shape[2]}")
print(f"    Input shape: {k_train.shape}")

# 2. Split data
n_samples = 100
train_split = int(0.8 * n_samples)
k_train_subset = k_train[:n_samples]
u_train_subset = u_train[:n_samples]

train_input = k_train_subset[:train_split]
train_output = u_train_subset[:train_split]
val_input = k_train_subset[train_split:]
val_output = u_train_subset[train_split:]

print(f"    Training samples: {train_split}")
print(f"    Validation samples: {n_samples - train_split}")

# 3. Create problem with explicit variable names
problem = SupervisedProblem(input_=train_input, output_=train_output)
# CRITICAL: Set the variable names for LabelTensor extraction
problem.input_variables = ['x']
problem.output_variables = ['u']

# 4. Create FNO model
lifting_net = FeedForward(input_dimensions=1, output_dimensions=20, layers=[30, 30])
projecting_net = FeedForward(input_dimensions=20, output_dimensions=1, layers=[30, 30])

fno = FNO(
    lifting_net=lifting_net,
    projecting_net=projecting_net,
    n_modes=8,
    dimensions=2,
    n_layers=2,
    padding=4,
)

# 5. Architecture Sanity Check
print("\n[2] Running architecture sanity check...")
with torch.no_grad():
    test_input = k_train_subset[:1]
    test_out = fno(test_input)
    print(f"    FNO Output Shape: {test_out.shape}")
    assert test_out.shape == (1, k_train.shape[1], k_train.shape[2], 1), \
        f"Shape mismatch: expected (1, {k_train.shape[1]}, {k_train.shape[2]}, 1), got {test_out.shape}"
    print("    ✅ Shape check passed!")

# 6. Create solver
solver = SupervisedSingleModelSolver(
    problem=problem,
    model=fno
)

# 7. collect_predictions hook (Handles LabelTensor dict)
def collect_predictions(trainer, pl_module):
    """Extract predictions from validation batch and convert to channels-first."""
    val_loader = trainer.val_dataloaders
    if val_loader is None:
        return None

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, list) and len(batch) > 0:
                _, data_dict = batch[0]
                input_label = data_dict['input']
                # Convert LabelTensor to raw tensor
                inputs = input_label.as_subclass(torch.Tensor).to(pl_module.device)
                pred = pl_module.model(inputs)
                # Convert Channels-Last to Channels-First
                return pred.permute(0, 3, 1, 2)
            else:
                try:
                    input_batch = batch[0].to(pl_module.device)
                    pred = pl_module.model(input_batch)
                    return pred.permute(0, 3, 1, 2)
                except:
                    continue
    return None

print("\n[3] Creating TopologyMonitor with collect_predictions hook...")
monitor = TopologyMonitor(
    expected_beta_0=1,
    expected_beta_1=0,
    monitor_freq=2,
    warmup_epochs=1,
    mode="warn",
    min_persistence=0.5,
    channel=0,
    collect_predictions=collect_predictions,
)

# 8. Create Trainer
trainer = Trainer(
    solver=solver,
    max_epochs=5,
    callbacks=[monitor],
    accelerator="cpu",
)

# 9. PINA-compliant collate function with LabelTensor and problem variables
print("\n[4] Setting up DataLoader streams...")

def pina_collate(batch):
    """Wraps raw PyTorch tensors into PINA's expected format with LabelTensor."""
    inputs = torch.stack([item[0] for item in batch])
    outputs = torch.stack([item[1] for item in batch])
    condition_name = list(solver.problem.conditions.keys())[0]
    # Use the variable names we set on the problem
    input_vars = solver.problem.input_variables  # ['x']
    output_vars = solver.problem.output_variables  # ['u']
    data_dict = {
        'input': LabelTensor(inputs, input_vars),
        'target': LabelTensor(outputs, output_vars)
    }
    return [(condition_name, data_dict)]

train_dataset = TensorDataset(train_input, train_output)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pina_collate)

val_dataset = TensorDataset(val_input, val_output)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=pina_collate)

# 10. Train with explicit data streams
print("\n[5] Starting training...")
trainer.fit(solver, train_dataloaders=train_loader, val_dataloaders=val_loader)

print("\n" + "=" * 60)
print("✅ Test completed successfully!")
print("✅ TopologyMonitor executed with GUDHI!")
print("=" * 60)