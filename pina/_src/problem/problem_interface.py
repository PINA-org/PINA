"""Module for the Problem Interface."""

from abc import ABCMeta, abstractmethod


class ProblemInterface(metaclass=ABCMeta):
    """
    Abstract interface for all problems.
    """

    @abstractmethod
    def __deepcopy__(self, memo):
        """
        Create a deep copy of the problem instance.

        :param dict[int, Any] memo: The object-id mapping maintained by
            :func:`copy.deepcopy` to preserve shared objects and cycles.
        :return: A deep copy of the problem instance.
        :rtype: ProblemInterface
        """

    @abstractmethod
    def discretise_domain(
        self, n=None, mode="random", domains=None, sample_rules=None
    ):
        """
        Discretise the problem's domains by sampling a specified number of
        points according to the selected sampling mode.

        :param n: The positive sample count or per-axis resolution passed to
            domain samplers. Ignored when ``sample_rules`` is provided. ``None``
            requires custom rules. Default is ``None``.
        :type n: int | None
        :param str mode: The sampling method. Available modes include:
            ``"random"`` for random sampling, ``"latin"`` or ``"lh"`` for latin
            hypercube sampling, ``"chebyshev"`` for Chebyshev sampling, and
            ``"grid"`` for grid sampling. Default is ``"random"``.
        :param domains: The domains from which to sample. If ``None``, all
            domains are considered for sampling. Default is ``None``.
        :type domains: str | list[str] | None
        :param sample_rules: The dictionary specifying custom sampling
            rules for each input variable. When provided, it overrides the
            global ``n`` and ``mode`` arguments. The dictionary keys must
            match every name in ``input_variables``, and
            each value must be a dictionary containing two keys: ``n`` for the
            number of points to sample for that variable, and ``mode`` for the
            sampling method to use. If ``None``, the global ``n`` and ``mode``
            parameters are used for all variables. Separate variable samples
            are combined as a Cartesian product. Default is ``None``.
        :type sample_rules: dict[str, dict[str, int | str]] | None

        .. warning::
            ``"random"`` is the only supported ``mode`` across all geometries:
            :class:`~pina.domain.CartesianDomain`,
            :class:`~pina.domain.EllipsoidDomain`, and
            :class:`~pina.domain.SimplexDomain`.
            Sampling modes such as ``"latin"``, ``"chebyshev"``, and ``"grid"``
            are only implemented for
            :class:`~pina.domain.CartesianDomain`. Custom rules require each
            domain to support the requested variablewise sampling; they do not
            preserve coordinate coupling in curved domains.
        """

    @abstractmethod
    def add_points(self, new_points_dict):
        """
        Append additional points to an already discretised domain.

        :param dict[str, LabelTensor] new_points_dict: The mapping of already
            sampled domain names to additional points, with compatible shapes
            and ordered labels.
        """

    @abstractmethod
    def move_discretisation_into_conditions(self):
        """
        Move the sampled points from the discretised domains into their
        corresponding conditions. This ensures that the conditions are evaluated
        on the correct set of points after discretisation.

        The trainer invokes this operation internally before training.
        """

    @property
    @abstractmethod
    def input_variables(self):
        """
        The input variables of the problem.

        :return: The input variables of the problem.
        :rtype: list[str]
        """

    @property
    @abstractmethod
    def output_variables(self):
        """
        The output variables of the problem.

        :return: The output variables of the problem.
        :rtype: list[str]
        """

    @property
    @abstractmethod
    def conditions(self):
        """
        The conditions associated with the problem.

        :return: The conditions associated with the problem.
        :rtype: dict[str, pina.condition.BaseCondition]
        """

    @property
    @abstractmethod
    def discretised_domains(self):
        """
        The dictionary containing the discretised domains of the problem. Each
        key corresponds to a domain name in ``domains``, and each value
        is a :class:`~pina.LabelTensor` containing the sampled points for
        that domain.

        :return: The discretised domains.
        :rtype: dict[str, LabelTensor]
        """

    @property
    @abstractmethod
    def are_all_domains_discretised(self):
        """
        Whether all domains of the problem have been discretised.

        :return: ``True`` if all domains are discretised, ``False`` otherwise.
        :rtype: bool
        """
