"""
Integration test for TopologyMonitor using synthetic Darcy-like data.
"""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from pina.model import FNO, FeedForward
from pina.solver import SupervisedSingleModelSolver
from pina.problem.zoo import SupervisedProblem
from pina import Trainer, LabelTensor
from pina._src.callback.topology.topology_monitor import TopologyMonitor

# Skip test if GUDHI is not installed
try:
    import gudhi  # noqa: F401
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False


def generate_synthetic_darcy(n_samples=20, grid_size=12):
    """
    Generate synthetic Darcy-like data for testing.

    FNO expects Channels-Last: (Batch, Height, Width, Channels).
    Returns (n_samples, grid_size, grid_size, 1).
    """
    x = torch.linspace(0, 1, grid_size)
    y = torch.linspace(0, 1, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    inputs = []
    outputs = []
    for _ in range(n_samples):
        k = 0.5 + 0.5 * torch.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.1)
        k = k + 0.05 * torch.randn(grid_size, grid_size)
        u = 1.0 - k
        inputs.append(k.unsqueeze(-1))
        outputs.append(u.unsqueeze(-1))

    inputs = torch.stack(inputs)   # (n_samples, grid_size, grid_size, 1)
    outputs = torch.stack(outputs) # (n_samples, grid_size, grid_size, 1)
    return inputs, outputs


def pina_collate(batch):
    """
    Collate function that structures data for PINA solvers.
    """
    inputs = torch.stack([item[0] for item in batch])
    outputs = torch.stack([item[1] for item in batch])
    data_dict = {
        'input': LabelTensor(inputs, ['x']),
        'target': LabelTensor(outputs, ['u'])
    }
    return [('data', data_dict)]


@pytest.mark.skipif(not GUDHI_AVAILABLE, reason="GUDHI not installed")
def test_topology_monitor_integration():
    """
    Integration test for TopologyMonitor with synthetic data.
    """
    print("=" * 60)
    print("Testing TopologyMonitor with synthetic Darcy-like data")
    print("=" * 60)

    # Generate synthetic data
    inputs, outputs = generate_synthetic_darcy(n_samples=20, grid_size=12)
    train_split = 14
    train_input = inputs[:train_split]
    train_output = outputs[:train_split]
    val_input = inputs[train_split:]
    val_output = outputs[train_split:]

    # Wrap in LabelTensor
    train_input_lt = LabelTensor(train_input, ['x'])
    train_output_lt = LabelTensor(train_output, ['u'])

    # Create problem
    problem = SupervisedProblem(input_=train_input_lt, output_=train_output_lt)
    problem.input_variables = ['x']
    problem.output_variables = ['u']

    # Tiny FNO for fast testing
    lifting_net = FeedForward(input_dimensions=1, output_dimensions=8, layers=[8, 8])
    projecting_net = FeedForward(input_dimensions=8, output_dimensions=1, layers=[8, 8])
    fno = FNO(
        lifting_net=lifting_net,
        projecting_net=projecting_net,
        n_modes=4,
        dimensions=2,
        n_layers=1,
        padding=2,
        inner_size=8,
    )
    solver = SupervisedSingleModelSolver(problem=problem, model=fno)

    # Data loaders with num_workers=0
    train_loader = DataLoader(
        TensorDataset(train_input, train_output),
        batch_size=8,
        shuffle=True,
        collate_fn=pina_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_input, val_output),
        batch_size=8,
        shuffle=False,
        collate_fn=pina_collate,
        num_workers=0,
    )

    def custom_extractor(batch, batch_idx, outputs):
        try:
            return batch[0][1]['input']
        except (IndexError, KeyError, TypeError):
            return None

    # Create monitor
    monitor = TopologyMonitor(
        expected_beta_0=1,
        expected_beta_1=0,
        monitor_freq=1,
        warmup_epochs=0,
        mode="warn",
        min_persistence=0.5,
        collect_predictions=custom_extractor,
    )

    # Trainer with solver in constructor
    trainer = Trainer(
        solver=solver,
        max_epochs=1,
        callbacks=[monitor],
        accelerator="cpu",
    )
    # CRITICAL: fit() expects 'model' as first positional argument (the solver)
    trainer.fit(solver, train_dataloaders=train_loader, val_dataloaders=val_loader)

    assert hasattr(monitor, 'profiler'), "Monitor missing profiler"
    assert trainer.current_epoch >= 0, "Trainer did not run"

    print("=" * 60)
    print("✅ TopologyMonitor integration test passed!")
    print("=" * 60)


def test_topology_monitor_instantiation():
    """Test that TopologyMonitor can be instantiated with valid parameters."""
    monitor = TopologyMonitor(expected_beta_0=1, mode="warn", warmup_epochs=10)
    assert monitor.expected_beta_0 == 1
    assert monitor.mode == "warn"
    assert monitor.warmup_epochs == 10
    print("✅ TopologyMonitor instantiation test passed!")


if __name__ == "__main__":
    test_topology_monitor_instantiation()
    if GUDHI_AVAILABLE:
        test_topology_monitor_integration()
    else:
        print("⚠️ GUDHI not installed. Skipping integration test.")