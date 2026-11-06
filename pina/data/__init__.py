"""Backward-compatible imports for data module utilities."""

import warnings

from pina._src.core.data_module import DataModule as _DataModule

__all__ = ["DataModule"]

_DEPRECATED_IMPORTS = {
    "DataModule": _DataModule,
    "PinaDataModule": _DataModule,
}


def __getattr__(name):
    """
    Resolve a deprecated data-module import and warn about its replacement.

    :param str name: The requested attribute name. ``"DataModule"`` and
        ``"PinaDataModule"`` resolve to :class:`pina.DataModule` and emit a
        :class:`DeprecationWarning`.
    :raises AttributeError: If ``name`` is not a supported compatibility alias.
    :return: The PINA data-module class.
    :rtype: type[pina.DataModule]
    """
    if name in _DEPRECATED_IMPORTS:
        warnings.warn(
            f"Importing '{name}' from 'pina.data' is deprecated; "
            "use pina.DataModule instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_IMPORTS[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
