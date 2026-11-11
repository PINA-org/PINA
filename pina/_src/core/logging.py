"""Logging utilities for PINA.

PINA uses the standard :mod:`logging` module and exposes a single ``pina``
logger namespace. Libraries should not configure the root logger, so this
module only installs a :class:`logging.NullHandler` at import time. Users can
opt in to console output with :func:`enable_debug_logging` or by setting the
``PINA_LOG_LEVEL`` environment variable.

:Example:

    >>> import pina
    >>> pina.enable_debug_logging()
    >>> pina.logger.debug("This message is now visible.")

    >>> # Alternatively, configure logging before importing PINA.
    >>> # PINA_LOG_LEVEL=DEBUG python train.py
"""

import logging
import os
import warnings

DEFAULT_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
"""Default format used by :func:`enable_debug_logging`."""

DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
"""Default date format used by :func:`enable_debug_logging`."""

LOG_LEVEL_ENV_VAR = "PINA_LOG_LEVEL"
"""Environment variable used to configure the PINA logging level."""

LOG_FORMAT_ENV_VAR = "PINA_LOG_FORMAT"
"""Environment variable used to configure the PINA logging format."""

_HANDLER_ATTRIBUTE = "_pina_handler"

logger = logging.getLogger("pina")
"""The root logger of the PINA logging hierarchy."""

# A NullHandler keeps libraries silent by default. It also prevents the
# ``logging.lastResort`` handler from writing to stderr.
if not any(
    isinstance(handler, logging.NullHandler) for handler in logger.handlers
):
    logger.addHandler(logging.NullHandler())


def _resolve_level(level):
    """
    Resolve a logging level from an integer or a level name.

    :param level: The logging level. It can be an integer, such as
        ``logging.DEBUG``, or a case-insensitive string, such as ``"DEBUG"``.
    :type level: int | str
    :raises TypeError: If ``level`` is neither an integer nor a string.
    :raises ValueError: If ``level`` is a string that does not name a
        standard logging level.
    :return: The resolved logging level.
    :rtype: int
    """
    if isinstance(level, int):
        return level

    if not isinstance(level, str):
        raise TypeError(
            "The logging level must be an integer or a string, "
            f"but got {type(level).__name__}."
        )

    resolved = logging.getLevelName(level.upper())
    if not isinstance(resolved, int):
        raise ValueError(  # noqa: TRY004 - invalid level name, not type
            f"Unknown logging level {level!r}. Use an integer or one of "
            "'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'."
        )

    return resolved


def get_logger(name=None):
    """
    Return a logger in the PINA ``pina`` namespace.

    Module level loggers should be created with ``get_logger(__name__)``.
    Names below the internal ``pina._src`` package are shortened so that, for
    example, ``pina._src.core.data_module`` becomes ``pina.core.data_module``.

    :param str name: The logger name. If ``None`` or ``"pina"``, the root PINA
        logger is returned. Default is ``None``.
    :return: The requested logger.
    :rtype: logging.Logger

    :Example:

        >>> from pina import get_logger
        >>> logger = get_logger("pina._src.core.data_module")
        >>> logger.name
        'pina.core.data_module'
    """
    if name is None or name == "pina":
        return logger

    if name.startswith("pina._src."):
        name = f"pina.{name[len('pina._src.'):]}"
    elif name == "pina._src":
        name = "pina"
    elif not name.startswith("pina."):
        name = f"pina.{name}"

    return logging.getLogger(name)


def _get_pina_handler():
    """
    Return the console handler installed by PINA, if any.

    :return: The PINA handler or ``None`` when logging is not configured.
    :rtype: logging.Handler | None
    """
    for handler in logger.handlers:
        if getattr(handler, _HANDLER_ATTRIBUTE, False):
            return handler
    return None


def _ensure_pina_handler(stream=None):
    """
    Install the PINA console handler if it is missing.

    The handler is tagged so that repeated calls never install duplicates.

    :param stream: The stream used by the handler. If ``None``, the handler
        uses the standard error stream. Default is ``None``.
    :type stream: io.TextIOBase | None
    :return: The installed handler.
    :rtype: logging.Handler
    """
    handler = _get_pina_handler()

    if handler is None:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT)
        )
        setattr(handler, _HANDLER_ATTRIBUTE, True)
        logger.addHandler(handler)
    elif stream is not None and handler.stream is not stream:
        handler.setStream(stream)

    return handler


def set_level(level):
    """
    Set the logging level of the PINA logger.

    :param level: The new logging level, either an integer or a level name.
    :type level: int | str
    :raises TypeError: If ``level`` is neither an integer nor a string.
    :raises ValueError: If ``level`` is not a valid logging level name.
    :return: The configured PINA logger.
    :rtype: logging.Logger

    :Example:

        >>> import pina
        >>> _ = pina.set_level("INFO")
    """
    logger.setLevel(_resolve_level(level))
    return logger


def enable_debug_logging(stream=None):
    """
    Enable debug logging on the console.

    This installs a single console handler on the ``pina`` logger and sets its
    level to ``DEBUG``. Calling the function multiple times is safe: the
    handler is reused instead of being duplicated.

    :param stream: The stream used for logging output. If ``None``, standard
        error is used. Default is ``None``.
    :type stream: io.TextIOBase | None
    :return: The configured PINA logger.
    :rtype: logging.Logger

    :Example:

        >>> import pina
        >>> pina.enable_debug_logging()
        >>> pina.logger.debug("Debug output is now enabled.")
    """
    _ensure_pina_handler(stream)
    logger.setLevel(logging.DEBUG)
    return logger


def disable_logging():
    """
    Disable PINA logging.

    This removes the console handler installed by PINA and sets the logger
    level above ``CRITICAL``. Handlers added by users are left untouched.

    :return: The configured PINA logger.
    :rtype: logging.Logger

    :Example:

        >>> import pina
        >>> pina.disable_logging()
    """
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_ATTRIBUTE, False):
            logger.removeHandler(handler)
            handler.close()

    logger.setLevel(logging.CRITICAL + 1)
    return logger


def _configure_from_environment():
    """
    Configure PINA logging from environment variables.

    ``PINA_LOG_LEVEL`` sets the logger level and installs the console handler.
    ``PINA_LOG_FORMAT`` overrides the message format. Invalid levels are
    ignored with a :class:`RuntimeWarning`.
    """
    level_env = os.environ.get(LOG_LEVEL_ENV_VAR)
    format_env = os.environ.get(LOG_FORMAT_ENV_VAR)

    if level_env is None and format_env is None:
        return

    level = None
    if level_env is not None:
        try:
            level = _resolve_level(level_env)
        except (TypeError, ValueError) as error:
            warnings.warn(
                f"Ignoring invalid {LOG_LEVEL_ENV_VAR}={level_env!r}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

    if level is None and format_env is None:
        return

    handler = _ensure_pina_handler()
    if format_env is not None:
        handler.setFormatter(
            logging.Formatter(format_env, DEFAULT_DATE_FORMAT)
        )

    if level is not None:
        logger.setLevel(level)


_configure_from_environment()
