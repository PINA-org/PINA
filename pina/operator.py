"""A public API for differential operators and automatic differentiation
utilities.

This module provides standard vector calculus operators (gradient, divergence,
laplacian, advection) implemented using automatic differentiation. It includes
both high-level general operators and optimized 'fast' variants for improved
computational efficiency during training.
"""

from pina._src.core.operator import (
    grad,
    fast_grad,
    fast_div,
    fast_laplacian,
    fast_advection,
    div,
    laplacian,
    advection,
)

__all__ = [
    "grad",
    "fast_grad",
    "fast_div",
    "fast_laplacian",
    "fast_advection",
    "div",
    "laplacian",
    "advection",
]
