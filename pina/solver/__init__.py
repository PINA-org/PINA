"""Module for all solvers in PINA."""

__all__ = [
    "SolverInterface",
    "BaseSolver",
    "SingleModelSolver",
    "MultiModelSolver",
    "EnsembleSolver",
    "SupervisedSingleModelSolver",
    "PhysicsInformedSingleModelSolver",
    "SupervisedEnsembleSolver",
    "PhysicsInformedEnsembleSolver",
    "AutoregressiveSingleModelSolver",
    "AutoregressiveEnsembleSolver",
    "SelfAdaptivePhysicsInformedSolver",
    "CompetitivePhysicsInformedSolver",
    "GradientPhysicsInformedSingleModelSolver",
    "RBAPhysicsInformedSingleModelSolver",
    "CausalPhysicsInformedSingleModelSolver",
]


from pina._src.solver.solver_interface import SolverInterface
from pina._src.solver.base_solver import BaseSolver
from pina._src.solver.single_model_solver import SingleModelSolver
from pina._src.solver.multi_model_solver import MultiModelSolver
from pina._src.solver.ensemble_solver import EnsembleSolver
from pina._src.solver.supervised_single_model_solver import (
    SupervisedSingleModelSolver,
)
from pina._src.solver.physics_informed_single_model_solver import (
    PhysicsInformedSingleModelSolver,
)
from pina._src.solver.supervised_ensemble_solver import SupervisedEnsembleSolver
from pina._src.solver.physics_informed_ensemble_solver import (
    PhysicsInformedEnsembleSolver,
)
from pina._src.solver.autoregressive_single_model_solver import (
    AutoregressiveSingleModelSolver,
)
from pina._src.solver.autoregressive_ensemble_solver import (
    AutoregressiveEnsembleSolver,
)
from pina._src.solver.self_adaptive_physics_informed_solver import (
    SelfAdaptivePhysicsInformedSolver,
)
from pina._src.solver.competitive_physics_informed_solver import (
    CompetitivePhysicsInformedSolver,
)
from pina._src.solver.gradient_physics_informed_single_model_solver import (
    GradientPhysicsInformedSingleModelSolver,
)
from pina._src.solver.rba_physics_informed_single_model_solver import (
    RBAPhysicsInformedSingleModelSolver,
)
from pina._src.solver.causal_physics_informed_single_model_solver import (
    CausalPhysicsInformedSingleModelSolver,
)

# Back-compatibility with version 0.2, to be removed soon
import warnings

_DEPRECATED_IMPORTS = {
    "SingleSolverInterface": "SingleModelSolver",
    "MultiSolverInterface": "MultiModelSolver",
    "DeepEnsembleSolverInterface": "EnsembleSolver",
    "SupervisedSolver": "SupervisedSingleModelSolver",
    "DeepEnsembleSupervisedSolver": "SupervisedEnsembleSolver",
    "PINN": "PhysicsInformedSingleModelSolver",
    "DeepEnsemblePINN": "PhysicsInformedEnsembleSolver",
    "GradientPINN": "GradientPhysicsInformedSingleModelSolver",
    "RBAPINN": "RBAPhysicsInformedSingleModelSolver",
    "CausalPINN": "CausalPhysicsInformedSingleModelSolver",
    "CompetitivePINN": "CompetitivePhysicsInformedSolver",
    "SelfAdaptivePINN": "SelfAdaptivePhysicsInformedSolver",
}

_REMOVED_IMPORTS = {
    "SupervisedSolverInterface": (
        "`SupervisedSolverInterface` has been removed. Its logic is now managed"
        " by `pina.solver.BaseSolver`, from which every solver inherits."
    ),
    "PINNInterface": (
        "`PINNInterface` has been removed. The underlying physics-informed "
        "logic is now handled by `pina.solver.mixin.PhysicsInformedMixin`."
    ),
    "ReducedOrderModelSolver": (
        "`ReducedOrderModelSolver` is no longer supported."
    ),
    "GAROM": ("`GAROM` is no longer supported."),
}


def __getattr__(name):
    """
    Resolve deprecated solver aliases and reject removed names.

    :param str name: The requested public solver attribute.
    :raises ImportError: If the requested solver/interface has been removed.
    :raises AttributeError: If no solver or alias has the requested name.
    :return: The current solver class for a recognized legacy alias, emitting
        ``DeprecationWarning``.
    :rtype: type
    """
    if name in _DEPRECATED_IMPORTS:

        warnings.warn(
            f"Importing '{name}' from 'pina.solver' is deprecated; use "
            f"pina.solver.{_DEPRECATED_IMPORTS[name]} instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return globals()[_DEPRECATED_IMPORTS[name]]

    if name in _REMOVED_IMPORTS:
        raise ImportError(_REMOVED_IMPORTS[name])

    raise AttributeError(f"module 'pina.solver' has no attribute '{name}'")
