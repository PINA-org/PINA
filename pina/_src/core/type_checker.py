"""Module for enforcing type hints in Python functions."""

import inspect
import logging
import typing

from pina._src.core.logging import get_logger

logger = get_logger(__name__)


def enforce_types(func):
    """
    Function decorator to enforce type hints at runtime.

    This decorator checks the types of the arguments and of the return value of
    the decorated function against the type hints specified in the function
    signature. If the types do not match, a TypeError is raised.
    Type checking is only performed when the logging level is set to `DEBUG`.

    :param Callable func: The function to be decorated.
    :return: The decorated function with enforced type hints.
    :rtype: Callable

    :Example:

    >>> @enforce_types
        def dummy_function(a: int, b: float) -> float:
        ... return a+b

        # This always works.
        dummy_function(1, 2.0)

        # This raises a TypeError for the second argument, if logging is set to
        # `DEBUG`.
        dummy_function(1, "Hello, world!")


    >>> @enforce_types
        def dummy_function2(a: int, right: bool) -> float:
        ... if right:
        ...     return float(a)
        ... else:
        ...     return "Hello, world!"

        # This always works.
        dummy_function2(1, right=True)

        # This raises a TypeError for the return value if logging is set to
        # `DEBUG`.
        dummy_function2(1, right=False)
    """

    logger.debug("enforce_types: wrapping %s.", func.__qualname__)

    def wrapper(*args, **kwargs):
        """
        Wrapper function to enforce type hints.

        :param tuple args: Positional arguments passed to the function.
        :param dict kwargs: Keyword arguments passed to the function.
        :raises TypeError: If the argument or return type does not match the
            specified type hints.
        :return: The result of the decorated function.
        :rtype: Any
        """
        level = logging.getLevelName(get_logger().getEffectiveLevel())

        # Enforce type hints only in debug mode
        if level != "DEBUG":
            return func(*args, **kwargs)

        logger.debug("enforce_types: checking %s.", func.__qualname__)

        # Get the type hints for the function arguments
        hints = typing.get_type_hints(func)
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        for arg_name, arg_value in bound.arguments.items():
            expected_type = hints.get(arg_name)
            if expected_type and not isinstance(arg_value, expected_type):
                logger.debug(
                    "enforce_types: argument '%s' of %s expected %s, got %s.",
                    arg_name,
                    func.__qualname__,
                    expected_type.__name__,
                    type(arg_value).__name__,
                )
                raise TypeError(
                    f"Argument '{arg_name}' must be {expected_type.__name__}, "
                    f"but got {type(arg_value).__name__}!"
                )

        # Get the type hints for the return values
        return_type = hints.get("return")
        result = func(*args, **kwargs)

        if return_type and not isinstance(result, return_type):
            logger.debug(
                "enforce_types: return value of %s expected %s, got %s.",
                func.__qualname__,
                return_type.__name__,
                type(result).__name__,
            )
            raise TypeError(
                f"Return value must be {return_type.__name__}, "
                f"but got {type(result).__name__}!"
            )

        return result

    return wrapper
