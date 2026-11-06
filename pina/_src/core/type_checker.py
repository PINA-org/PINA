"""Module for enforcing type hints in Python functions."""

import inspect
import typing
import logging


def enforce_types(func):
    """
    Function decorator to enforce type hints at runtime.

    This decorator checks the types of the arguments and of the return value of
    the decorated function against the type hints specified in the function
    signature using :func:`isinstance`. Checking is performed at call time only
    when the root logger's effective level has the name ``"DEBUG"``. At other
    levels, arguments and results are passed through without annotation checks.

    Default argument values are checked after signature binding. Each bound
    argument is checked as a whole; container elements and individual variadic
    arguments are not checked recursively. Annotations must be supported by
    ``isinstance``. Parameterized generics such as ``list[int]`` and
    ``typing.Any`` are not supported.

    :param Callable func: The function to be decorated.
    :return: The wrapper, which returns the original function's result and
        raises on type mismatches while debug checking is active. Function
        metadata and the original signature are not copied onto the wrapper.
    :rtype: Callable

    :Example:

        >>> import logging
        >>> from pina.type_checker import enforce_types
        >>> @enforce_types
        ... def add(a: int, b: float) -> float:
        ...     return a + b
        >>> root = logging.getLogger()
        >>> previous_level = root.level
        >>> root.setLevel(logging.DEBUG)
        >>> try:
        ...     print(add(1, 2.0))
        ...     add(1, 2)
        ... except TypeError:
        ...     print("Type mismatch")
        ... finally:
        ...     root.setLevel(previous_level)
        3.0
        Type mismatch

    .. warning::

        Mismatch reporting accesses the annotation's ``__name__``. Some
        annotations, including PEP 604 unions, lack this attribute and raise
        :class:`AttributeError` instead of the intended :class:`TypeError`.
    """

    def wrapper(*args, **kwargs):
        """
        Wrapper function to enforce type hints.

        :param tuple[Any, ...] args: Positional arguments passed to the original
            function and bound against its signature in debug mode.
        :param dict[str, Any] kwargs: Keyword arguments passed to the original
            function and bound against its signature in debug mode.
        :raises TypeError: If signature binding fails, a supported annotation
            does not match an argument or return value in debug mode, or an
            annotation cannot be used with :func:`isinstance`.
        :raises NameError: If debug-mode type-hint resolution encounters an
            unresolved forward reference.
        :raises AttributeError: If debug-mode mismatch reporting accesses an
            annotation without ``__name__``.
        :return: The result of the decorated function.
        :rtype: Any
        """
        level = logging.getLevelName(logging.getLogger().getEffectiveLevel())

        # Enforce type hints only in debug mode
        if level != "DEBUG":
            return func(*args, **kwargs)

        # Get the type hints for the function arguments
        hints = typing.get_type_hints(func)
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        for arg_name, arg_value in bound.arguments.items():
            expected_type = hints.get(arg_name)
            if expected_type and not isinstance(arg_value, expected_type):
                raise TypeError(
                    f"Argument '{arg_name}' must be {expected_type.__name__}, "
                    f"but got {type(arg_value).__name__}!"
                )

        # Get the type hints for the return values
        return_type = hints.get("return")
        result = func(*args, **kwargs)

        if return_type and not isinstance(result, return_type):
            raise TypeError(
                f"Return value must be {return_type.__name__}, "
                f"but got {type(result).__name__}!"
            )

        return result

    return wrapper
