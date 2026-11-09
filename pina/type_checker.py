"""Runtime type enforcement and validation utilities.

The decorator checks annotated arguments and return values with ``isinstance``
only when the root logger's effective level is ``DEBUG``. It is not a recursive
validator for parameterized container annotations.
"""

from pina._src.core.type_checker import enforce_types

__all__ = ["enforce_types"]
