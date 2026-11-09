"""
Mathematical equations and physical laws.

This module provides a framework for defining differential equations,
boundary conditions, and complex systems of equations. It includes
pre-defined physical models such as Poisson, Helmholtz, and Wave equations,
along with equations for common derivative-based constraints, such as
FixedValue, FixedGradient, FixedFlux, and FixedLaplacian.
"""

__all__ = [
    "EquationInterface",
    "BaseEquation",
    "Equation",
    "SystemEquation",
]

from pina._src.equation.equation_interface import EquationInterface
from pina._src.equation.base_equation import BaseEquation
from pina._src.equation.equation import Equation
from pina._src.equation.system_equation import SystemEquation

# Back-compatibility with version 0.2, to be removed soon
import warnings
import importlib

_DEPRECATED_IMPORTS = {
    "FixedValue": ".zoo",
    "FixedGradient": ".zoo",
    "FixedFlux": ".zoo",
    "FixedLaplacian": ".zoo",
    "Laplace": ".zoo",
    "HelmholtzEquation": ".zoo",
    "PoissonEquation": ".zoo",
    "AcousticWaveEquation": ".zoo",
    "AdvectionEquation": ".zoo",
    "AllenCahnEquation": ".zoo",
    "DiffusionReactionEquation": ".zoo",
    "BurgersEquation": ".zoo",
}


def __getattr__(name):
    """
    Resolve legacy equation-zoo imports with a deprecation warning.

    :param str name: The requested attribute name.
    :return: The matching class from :mod:`pina.equation.zoo`, or ``None`` for
        an unrecognized name. Recognized names emit ``DeprecationWarning``.
    :rtype: type | None
    """
    if name in _DEPRECATED_IMPORTS:

        warnings.warn(
            f"Importing '{name}' from 'pina.equation' is deprecated; "
            f"import it from 'pina.equation.zoo' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        module = importlib.import_module(_DEPRECATED_IMPORTS[name], __name__)
        return getattr(module, name)
