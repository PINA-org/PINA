"""Deprecated Condition factory."""

from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)
from pina._src.condition.graph_time_series_condition import (
    GraphTimeSeriesCondition,
)
from pina._src.condition.input_equation_condition import InputEquationCondition
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.condition.time_series_condition import TimeSeriesCondition


class Condition:
    """
    Deprecated factory for condition objects.

    Use the direct condition classes from :mod:`pina.condition` instead.

    Dispatch depends only on keyword names, not on the types of their values.
    Constructor validation is delegated to the selected condition class.
    """

    _SPECS = (
        (InputTargetCondition, {"input", "target"}, set()),
        (InputEquationCondition, {"input", "equation"}, set()),
        (DomainEquationCondition, {"domain", "equation"}, set()),
        (
            TimeSeriesCondition,
            {"input", "n_windows", "unroll_length"},
            {"randomize"},
        ),
        (
            GraphTimeSeriesCondition,
            {"input", "n_windows", "unroll_length"},
            {"key", "randomize"},
        ),
    )

    def __new__(cls, *args, **kwargs):
        """
        Construct the condition selected by the supplied keyword names.

        :param tuple[Any, ...] args: Positional arguments, which are rejected.
        :param dict[str, Any] kwargs: Keyword arguments forwarded unchanged to
            the selected constructor. Supported combinations are:

            * ``input`` and ``target``: Tensor samples passed to
              :class:`pina.condition.InputTargetCondition`.
            * ``input`` and ``equation``: Tensor samples and a
              :class:`pina.equation.BaseEquation`, passed to
              :class:`pina.condition.InputEquationCondition`.
            * ``domain`` and ``equation``: A domain object or registered name
              and an equation, passed to
              :class:`pina.condition.DomainEquationCondition`.
            * ``input``, ``n_windows``, and ``unroll_length``: Temporal tensors
              and integer window settings, passed to
              :class:`pina.condition.TimeSeriesCondition`. Optional
              ``randomize`` (bool) controls window randomization. Default is
              ``False``.

            Values are validated by the selected constructor. An additional
            ``key`` in the temporal combination selects the legacy graph branch
            described below rather than a working tensor signature.
        :raises ValueError: If positional arguments are supplied, the keyword
            combination is unsupported, or the selected constructor rejects
            argument values.
        :raises AssertionError: If temporal window counts or lengths fail the
            selected constructor's positive-integer validation.
        :raises TypeError: If the legacy graph branch is selected, because its
            forwarded keywords do not match the graph condition constructor.
        :return: The condition created by the selected constructor, not an
            instance of this factory class.
        :rtype: InputTargetCondition | InputEquationCondition |
            DomainEquationCondition | TimeSeriesCondition

        .. warning::

            The legacy graph branch cannot construct graph time-series
            conditions: without ``key``, tensor dispatch takes precedence;
            with ``key``, the forwarded arguments are incompatible. Construct
            :class:`pina.condition.GraphTimeSeriesCondition` directly instead.
        """
        if args:
            raise ValueError("Condition takes only keyword arguments.")

        for condition_cls, required, optional in cls._SPECS:
            allowed = required | optional
            if required <= set(kwargs) <= allowed:
                return condition_cls(**kwargs)

        valid_signatures = [
            sorted(required | optional) for _, required, optional in cls._SPECS
        ]
        raise ValueError(
            f"Invalid keyword arguments {sorted(set(kwargs))}. "
            f"Valid signatures are: {valid_signatures}."
        )
