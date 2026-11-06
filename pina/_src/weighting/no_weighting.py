"""Unit-weight aggregation of condition losses."""

from pina._src.weighting.scalar_weighting import ScalarWeighting


class _NoWeighting(ScalarWeighting):
    """
    Weighting strategy that leaves all loss terms unchanged.

    This is a special case of scalar weighting where a unit weight is assigned
    to every loss term, resulting in no reweighting.
    """

    def __init__(self):
        """
        Initialization of the :class:`_NoWeighting` class.
        """
        super().__init__(weights=1)
