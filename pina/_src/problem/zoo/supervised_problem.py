"""Formulation of a Supervised Problem in PINA."""

from pina._src.problem.base_problem import BaseProblem
from pina._src.condition.input_target_condition import InputTargetCondition


class SupervisedProblem(BaseProblem):
    """
    Definition of a supervised-learning problem.

    This class provides a simple way to define a supervised problem using the
    :class:`~pina.condition.input_target_condition.InputTargetCondition`.

    :Example:

        >>> import torch
        >>> input_data = torch.rand((100, 10))
        >>> output_data = torch.rand((100, 10))
        >>> problem = SupervisedProblem(input_data, output_data)
    """

    # TODO: This is necessary to override the abstract properties of
    # BaseProblem, but it is not an ideal solution. We should consider
    # a different desgin to manage input and output variables.
    conditions = {}
    output_variables = None
    input_variables = None

    def __init__(
        self, input_, output_, input_variables=None, output_variables=None
    ):
        """
        Initialization of the :class:`SupervisedProblem` class.

        :param input_: Input data of the problem.
        :type input_: torch.Tensor | LabelTensor | Graph | Data
        :param output_: Output data of the problem.
        :type output_: torch.Tensor | LabelTensor | Graph | Data
        :param list[str] input_variables: List of names of the input variables.
            If None, the input variables are inferred from `input_`.
            Default is ``None``.
        :param list[str] output_variables: List of names of the output
            variables. If None, the output variables are inferred from
            `output_`. Default is ``None``.
        """
        # Set input and output variables
        self.input_variables = input_variables
        self.output_variables = output_variables

        # Set the condition
        self.conditions["data"] = InputTargetCondition(input=input_, target=output_)
        super().__init__()
