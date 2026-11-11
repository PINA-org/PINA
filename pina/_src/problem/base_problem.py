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
    defined in :class:`~pina.problem.problem_interface.ProblemInterface`.

    This class is not meant to be instantiated directly.
    """

    def __init__(self):
        """
        Initialization of the :class:`BaseProblem` class.
        """
        self._discretised_domains = {}

        # Copy the conditions so class-level dictionaries are not mutated
        self.conditions = dict(self.conditions)

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

        :param dict memo: The memorization dictionary used by the deepcopy
            function.
        :return: A deep copy of the problem instance.
        :rtype: ProblemInterface
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

        :param int n: The number of points to sample. This is ignored if
            ``sample_rules`` is provided. Default is ``None``.
        :param str mode: The sampling method. Available modes include:
            ``"random"`` for random sampling, ``"latin"`` or ``"lh"`` for latin
            hypercube sampling, ``"chebyshev"`` for Chebyshev sampling, and
            ``"grid"`` for grid sampling. Default is ``"random"``.
        :param domains: The domains from which to sample. If ``None``, all
            domains are considered for sampling. Default is ``None``.
        :type domains: str | list[str]
        :param dict sample_rules: The dictionary specifying custom sampling
            rules for each input variable. When provided, it overrides the
            global ``n`` and ``mode`` arguments. Each key in the dictionary must
            match one of the variables defined in :meth:`input_variables`, and
            each value must be a dictionary containing two keys: ``n`` for the
            number of points to sample for that variable, and ``mode`` for the
            sampling method to use. If ``None``, the global ``n`` and ``mode``
            parameters are used for all variables. Default is ``None``.
        :raises ValueError: If ``sample_rules`` is provided but it is not a
            dictionary.
        :raises ValueError: If ``sample_rules`` is provided but its keys do not
            match the input variables of the problem.
        :raises ValueError: If ``sample_rules`` is provided but any of its rules
            is not a dictionary containing both ``n`` and ``mode`` keys, with
            ``n`` being a positive integer and ``mode`` being a string.
        :raises AssertionError: If ``n`` is not a positive integer.
        :raises ValueError: If ``mode`` is not a string
        :raises ValueError: If ``domains`` is provided by it is neither a string
            nor a list of strings.

        .. warning::
            ``"random"`` is the only supported ``mode`` across all geometries:
            :class:`~pina.domain.cartesian_domain.CartesianDomain`,
            :class:`~pina.domain.ellipsoid_domain.EllipsoidDomain`, and
            :class:`~pina.domain.simplex_domain.SimplexDomain`.
            Sampling modes such as ``"latin"``, ``"chebyshev"``, and ``"grid"``
            are only implemented for
            :class:`~pina.domain.cartesian_domain.CartesianDomain`.
            When custom discretisation is specified via ``sample_rules``, the
            domain to be discretised must be an instance of
            :class:`~pina.domain.cartesian_domain.CartesianDomain`.
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

        :param dict new_points_dict: The dictionary mapping each domain to the
            corresponding set of new points to be added. Each key in the
            dictionary must match one of the domains defined in :attr:`domains`,
            and each value must be a :class:`~pina.tensor.LabelTensor`
            containing the new points to be added to that domain. The labels of
            the points to be added must correspond to those of the domain to
            which they are being added.
        :raises ValueError: If ``new_points_dict`` is not a dictionary.
        :raises ValueError: If any of the values in ``new_points_dict`` is not
            a :class:`~pina.tensor.LabelTensor`.
        :raises ValueError: If any of the keys in ``new_points_dict`` does not
            match any of the domains defined in :attr:`domains`.
        :raises ValueError: If any of the domains in ``new_points_dict`` has not
            been discretised yet.
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

        :return: The input variables of the problem.
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
        key corresponds to a domain defined in :attr:`domains`, and each value
        is a :class:`~pina.tensor.LabelTensor` containing the sampled points for
        that domain.

        :return: The discretised domains.
        :rtype: dict
        """
        return self._discretised_domains

    @property
    def are_all_domains_discretised(self):
        """
        Whether all domains of the problem have been discretised.

        :return: ``True`` if all domains are discretised, ``False`` otherwise.
        :rtype: bool
        """
        return all(d in self.discretised_domains for d in self.domains)
