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
    """

    _SPECS = (
        (InputTargetCondition, {"input", "target"}, set()),
        (InputEquationCondition, {"input", "equation"}, set()),
        (DomainEquationCondition, {"domain", "equation"}, set()),
        (
            TimeSeriesCondition,
            {"input"},
            {"state", "control"},
        ),
        (
            GraphTimeSeriesCondition,
            {"input", "n_windows", "unroll_length"},
            {"key", "randomize"},
        ),
    )

    def __new__(cls, *args, **kwargs):
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
