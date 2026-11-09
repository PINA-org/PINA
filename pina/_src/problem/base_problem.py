"""Module for the BaseProblem class."""

import warnings
from copy import deepcopy
from pina._src.problem.problem_interface import ProblemInterface
from pina._src.domain.domain_interface import DomainInterface
from pina._src.core.label_tensor import LabelTensor
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)
from pina._src.condition.input_equation_condition import (
    InputEquationCondition,
)
from pina._src.core.utils import (
    check_consistency,
    check_positive_integer,
    merge_tensors,
)


class BaseProblem(ProblemInterface):
    """
    Base class for all problems, implementing common functionality.

    A problem is defined by core components, including input and output
    variables, a set of conditions to be satisfied, and optionally the domains
    on which these conditions are defined.

    All problems must inherit from this class and implement abstract methods
    defined in :class:`~pina.problem.ProblemInterface`.

    Initialization binds existing conditions to this instance and rewrites
    directly supplied condition domains as named entries in ``domains``.
    Existing condition and domain dictionaries are not copied. Class-level
    dictionaries and their condition objects can therefore be shared and
    modified by multiple problem instances.

    This class is not meant to be instantiated directly.
    """

    def __init__(self):
        """
        Initialization of the :class:`BaseProblem` class.
        """
        self._discretised_domains = {}

        # Create a correspondence between the problem and the conditions
        for condition_name in self.conditions:
            self.conditions[condition_name].problem = self
            self.conditions[condition_name].name = condition_name

        # Create a dictionary to store the domains of the problem
        if not hasattr(self, "domains"):
            self.domains = {}

        # Store all the domains object passed to the problem's conditions
        for name, cond in self.conditions.items():
            if isinstance(cond, DomainEquationCondition):
                if isinstance(cond.domain, DomainInterface):
                    self.domains[name] = cond.domain
                    cond.domain = name

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the problem instance.

        Only instance attributes are recursively copied, using the memo to
        preserve cycles. Attributes inherited from the class are not copied.

        :param dict[int, Any] memo: The object-id mapping maintained by
            :func:`copy.deepcopy`; this copy is added before recursion.
        :return: A new instance of the same class, without calling its
            constructor.
        :rtype: BaseProblem
        """
        # Create a new instance of the same class and store it in a dictionary
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result

        # Set the attributes of the new instance to deep copies of the original
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))

        return result

    def discretise_domain(
        self, n=None, mode="random", domains=None, sample_rules=None
    ):
        """
        Discretise the problem's domains by sampling a specified number of
        points according to the selected sampling mode.

        :param n: The positive sample count or per-axis resolution passed to
            domain samplers. Ignored when ``sample_rules`` is provided. ``None``
            is valid only with custom rules; otherwise validation fails.
            Default is ``None``.
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
            parameters are used for all variables. Each variable is sampled
            separately and results are combined as a Cartesian product; this
            does not preserve coupling between coordinates of curved domains.
            Extra rule fields are ignored. Default is ``None``.
        :type sample_rules: dict[str, dict[str, int | str]] | None
        :raises ValueError: If ``sample_rules`` is provided but it is not a
            dictionary.
        :raises ValueError: If ``sample_rules`` is provided but its keys do not
            match the input variables of the problem.
        :raises ValueError: If ``sample_rules`` is provided but any of its rules
            is not a dictionary containing both ``"n"`` and ``"mode"`` keys,
            or a rule mode is not a string.
        :raises AssertionError: If a global or per-rule count being used is not
            a positive integer.
        :raises ValueError: If ``mode`` is not a string, even with custom rules.
        :raises ValueError: If ``domains`` is provided but is neither a string
            nor a list of strings.
        :raises ValueError: If a sampler rejects a mode or variable selection.
        :raises KeyError: If a selected domain name is unknown.

        .. warning::
            ``"random"`` is the only supported ``mode`` across all geometries:
            :class:`~pina.domain.CartesianDomain`,
            :class:`~pina.domain.EllipsoidDomain`, and
            :class:`~pina.domain.SimplexDomain`.
            Sampling modes such as ``"latin"``, ``"chebyshev"``, and ``"grid"``
            are only implemented for
            :class:`~pina.domain.CartesianDomain`. Custom rules are not
            restricted by domain type, but every domain must support the
            requested variablewise sampling. Existing sampled entries are
            overwritten, and earlier entries can be changed before a later
            sampling failure.
        """
        # Initialize the domains to be discretised
        if domains is None:
            domains = list(self.domains)
        if not isinstance(domains, (list)):
            domains = [domains]

        # Check sampling rules
        if sample_rules is not None:
            check_consistency(sample_rules, dict)

            # Check that the keys of sample_rules match the input variables
            if sorted(list(sample_rules.keys())) != sorted(
                self.input_variables
            ):
                raise ValueError(
                    "The keys of the sample_rules dictionary must match the "
                    "input variables."
                )

            # Check that the rules for each variable are valid
            for var, rules in sample_rules.items():
                check_consistency(rules, dict)
                if "n" not in rules or "mode" not in rules:
                    raise ValueError(
                        f"Sampling rules for variable {var} must contain 'n' "
                        "and 'mode' keys."
                    )
                check_positive_integer(rules["n"], strict=True)
                check_consistency(rules["mode"], str)

        # Check n only if sample_rules is not provided
        else:
            check_positive_integer(n, strict=True)

        # Check consistency
        check_consistency(mode, str)
        check_consistency(domains, str)

        # If sample_rules is provided, apply custom discretisation
        if sample_rules is not None:
            for d in domains:

                # Discretise each variable according to its custom rules
                discretised_tensor = [
                    self.domains[d].sample(rules["n"], rules["mode"], var)
                    for var, rules in sample_rules.items()
                ]

                # Merge the discretised tensors into a single one for the domain
                self.discretised_domains[d] = merge_tensors(discretised_tensor)

        # Otherwise, apply the same n and mode to all specified domains
        else:
            for d in domains:
                self.discretised_domains[d] = self.domains[d].sample(n, mode)

    def add_points(self, new_points_dict):
        """
        Append additional points to an already discretised domain.

        :param dict[str, LabelTensor] new_points_dict: A mapping of domain names
            to new point tensors. Each key must identify an already sampled
            domain. Tensors must have compatible non-sample dimensions and
            matching ordered labels for labeled vertical stacking.
        :raises ValueError: If ``new_points_dict`` is not a dictionary.
        :raises ValueError: If any of the values in ``new_points_dict`` is not
            a :class:`~pina.LabelTensor`.
        :raises ValueError: If any of the keys in ``new_points_dict`` does not
            match any name in ``domains``.
        :raises ValueError: If any of the domains in ``new_points_dict`` has not
            been discretised yet.
        :raises ValueError: If the stacked tensors' labels do not match.
        :raises RuntimeError: If tensor shapes are incompatible for stacking.
        """
        # Check consistency
        check_consistency(new_points_dict, dict)

        # Check the keys and values of the dictionary
        for key, value in new_points_dict.items():
            check_consistency(value, LabelTensor)
            if key not in self.domains:
                raise ValueError(
                    f"Key {key} does not match any domain of the problem."
                )
            if key not in self.discretised_domains:
                raise ValueError(f"Domain {key} has not been discretised yet.")

        # Append the new points to the corresponding discretised domains
        for key, value in new_points_dict.items():
            self.discretised_domains[key] = LabelTensor.vstack(
                [self.discretised_domains[key], value]
            )

    def move_discretisation_into_conditions(self):
        """
        Move the sampled points from the discretised domains into their
        corresponding conditions. This ensures that the conditions are evaluated
        on the correct set of points after discretisation.

        The trainer calls this method internally. It shallow-copies the
        condition dictionary, then replaces domain-linked conditions with
        :class:`pina.condition.InputEquationCondition` objects referencing the
        sampled tensors. It preserves their equation, domain name, problem,
        and condition name, but does not copy arbitrary condition attributes.
        The sampled-domain mapping remains populated. Conditions without a
        domain are left unchanged.

        :raises ValueError: If a domain-equation condition's domain has not been
            sampled. Earlier conditions may already have been replaced.
        """
        # Copy the conditions so class-level dictionaries are not mutated
        self.conditions = dict(self.conditions)

        # Move the discretised domains into their corresponding conditions
        for name, cond in list(self.conditions.items()):
            domain = getattr(cond, "domain", None)
            if domain is None:
                continue

            if domain not in self.discretised_domains:
                if isinstance(cond, DomainEquationCondition):
                    raise ValueError(
                        f"Domain '{domain}' of condition '{name}' has not "
                        "been discretised."
                    )
                continue

            # Create a new condition with the discretised domain as input
            new_condition = InputEquationCondition(
                input=self.discretised_domains[domain],
                equation=cond.equation,
            )

            # Set the domain and problem attributes of the new condition
            new_condition.domain = domain
            new_condition.problem = self
            new_condition.name = name

            # Replace the old condition in the conditions dictionary
            self.conditions[name] = new_condition

    @property
    def input_variables(self):
        """
        The input variables of the problem.

        :return: Spatial, temporal, then parametric variable names concatenated
            in that order. Scalar string declarations are converted to lists;
            names are not globally sorted or deduplicated. Unknown inverse
            parameters are not included.
        :rtype: list[str]
        """
        # Define a helper function to convert a string to a list if needed
        _as_list = lambda x: [x] if isinstance(x, str) else x

        # Collect the spatial, temporal, and parametric variables
        variables = []
        if hasattr(self, "spatial_variables"):
            variables += _as_list(self.spatial_variables)
        if hasattr(self, "temporal_variables"):
            variables += _as_list(self.temporal_variables)
        if hasattr(self, "parameters"):
            variables += _as_list(self.parameters)

        return variables

    @property
    def discretised_domains(self):
        """
        The dictionary containing the discretised domains of the problem. Each
        key corresponds to a domain name in ``domains``, and each value
        is a :class:`~pina.LabelTensor` containing the sampled points for
        that domain.

        :return: The stored mapping of domain names to sampled tensors. Mutating
            it changes the problem's discretisation directly.
        :rtype: dict[str, LabelTensor]
        """
        return self._discretised_domains

    @property
    def are_all_domains_discretised(self):
        """
        Whether all domains of the problem have been discretised.

        :return: Whether every name in ``domains`` has a sampled mapping entry.
            Empty domain mappings return ``True``; tensor contents are not
            validated by this property.
        :rtype: bool
        """
        return all(d in self.discretised_domains for d in self.domains)
