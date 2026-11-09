"""Formulation of the inverse Poisson problem in a square domain."""

import warnings
import requests
import torch
from io import BytesIO

from pina._src.core.utils import custom_warning_format, check_consistency
from pina._src.domain.cartesian_domain import CartesianDomain
from pina._src.problem.inverse_problem import InverseProblem
from pina._src.problem.spatial_problem import SpatialProblem
from pina._src.equation.zoo.fixed_value import FixedValue
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.core.label_tensor import LabelTensor
from pina._src.equation.equation import Equation
from pina._src.core.operator import laplacian

warnings.formatwarning = custom_warning_format
warnings.filterwarnings("always", category=ResourceWarning)


def _load_tensor_from_url(url, labels, timeout=10):
    """
    Download a serialized labeled tensor and rewrap its detached tensor data.

    Request failures emit ``ResourceWarning`` and return ``None``.
    Deserialization and label errors are not caught. Loading uses
    ``weights_only=False``:
    only pass trusted URLs because pickle loading can execute arbitrary code.

    :param str url: The trusted URL of a PyTorch-serialized object providing a
        tensor-valued ``tensor`` attribute.
    :param labels: Labels for the resulting LabelTensor.
    :type labels: list[str] | tuple[str, ...]
    :param timeout: The request timeout in seconds, forwarded to
        :func:`requests.get`. Default is ``10``.
    :type timeout: int | float
    :raises AttributeError: If the loaded object has no tensor-valued ``tensor``
        attribute supporting detachment.
    :raises ValueError: If label validation rejects the supplied labels.
    :return: A labeled tensor with detached data, or ``None`` after a request
        failure. Deserialization errors propagate unchanged.
    :rtype: LabelTensor | None
    """
    # Try to download the tensor file from the given URL
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        tensor = torch.load(
            BytesIO(response.content), weights_only=False
        ).tensor.detach()
        return LabelTensor(tensor, labels)

    # If the request fails, issue a warning and return None
    except requests.exceptions.RequestException as e:
        warnings.warn(
            f"Could not download data for 'InversePoisson2DSquareProblem' "
            f"from '{url}'. Reason: {e}. Skipping data loading.",
            ResourceWarning,
        )
        return None


def laplace_equation(input_, output_, params_):
    """
    Implementation of the laplace equation.

    :param LabelTensor input_: Points of shape ``(n, 2)`` with ``"x"`` and
        ``"y"`` labels and gradient tracking enabled.
    :param LabelTensor output_: Predictions labeled ``"u"``, with an autograd
        connection supporting second derivatives.
    :param dict[str, torch.nn.Parameter] params_: Unknown parameters under
        ``"mu1"`` and ``"mu2"``, broadcastable with coordinate columns.
    :raises KeyError: If a required parameter name is absent.
    :raises ValueError: If coordinate extraction rejects a missing label.
    :raises RuntimeError: If differentiation fails or an operator label is
        absent.
    :raises TypeError: If an unused autograd gradient cannot be indexed.
    :return: The spatial Laplacian minus Gaussian forcing, of shape ``(n, 1)``.
    :rtype: LabelTensor
    """
    force_term = torch.exp(
        -2 * (input_.extract(["x"]) - params_["mu1"]) ** 2
        - 2 * (input_.extract(["y"]) - params_["mu2"]) ** 2
    )
    delta_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
    return delta_u - force_term


class InversePoisson2DSquareProblem(SpatialProblem, InverseProblem):
    r"""
    Implementation of the inverse two-dimensional Poisson problem on the square
    domain :math:`\Omega = [-2, 2] \times [-2, 2]`, with unknown parameter
    domain :math:`\Theta = [-1, 1] \times [-1, 1]`.

    The problem is governed by the parameterized Poisson equation

    .. math::

        \Delta u
        =
        \exp\left(
        -2(x - \mu_1)^2
        -2(y - \mu_2)^2
        \right),

    where :math:`u = u(x, y)` is the solution field and :math:`\mu_1, \mu_2` are
    unknown parameters controlling the forcing term.

    Homogeneous Dirichlet boundary conditions are imposed on the boundary of the
    domain:

    .. math::

        u(x, y) = 0, \qquad (x, y) \in \partial \Omega.

    The inverse problem aims to infer the unknown parameters :math:`\mu_1` and
    :math:`\mu_2` from solution data.

    The ``"data"`` condition is added only if the required files are downloaded
    successfully.

    :Example:

        >>> from pina.problem.zoo import InversePoisson2DSquareProblem
        >>> problem = InversePoisson2DSquareProblem(load=False)
    """

    output_variables = ["u"]
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    spatial_domain = CartesianDomain({"x": [x_min, x_max], "y": [y_min, y_max]})
    unknown_parameter_domain = CartesianDomain({"mu1": [-1, 1], "mu2": [-1, 1]})

    domains = {
        "D": spatial_domain,
        "boundary": spatial_domain.partial(),
    }

    conditions = {
        "D": DomainEquationCondition(
            domain="D", equation=Equation(laplace_equation)
        ),
        "boundary": DomainEquationCondition(
            domain="boundary", equation=FixedValue(0.0)
        ),
    }

    def __init__(self, load=True, data_size=1.0):
        """
        Initialization of the :class:`InversePoisson2DSquareProblem` class.

        :param bool load: Whether to download trusted tutorial input/target
            files and add a supervised condition when both downloads succeed.
            ``False`` skips requests but does not remove a previously installed
            class-shared data condition. Default is ``True``.
        :param float data_size: The fraction of the total data to use for the
            ``"data"`` condition. Takes the first ``int(n * data_size)`` input
            rows and the same target slice, without shuffling. ``1.0`` uses all
            rows and ``0.0`` adds an empty data condition after successful
            downloads. Validated even with ``load=False``. Default is ``1.0``.
        :raises ValueError: If ``load`` is not boolean, ``data_size`` is not a
            float in ``[0.0, 1.0]``, or loaded sample counts are incompatible.

        .. warning::

            Tutorial files are deserialized using ``weights_only=False``.
            Request failures are warned about and skipped, while loading and
            format errors propagate. Importing this module also replaces the
            global warning formatter and enables ``ResourceWarning`` messages.
        """
        super().__init__()

        # Check consistency
        check_consistency(load, bool)
        check_consistency(data_size, float)
        if not 0.0 <= data_size <= 1.0:
            raise ValueError(
                f"data_size must be in the range [0.0, 1.0], got {data_size}."
            )

        # Load data if requested
        if load:

            # Define URLs for input and output data
            input_url = (
                "https://github.com/PINA-org/PINA-Tutorials/raw/refs/heads/main"
                "/tutorials/tutorial7/data/pts_0.5_0.5"
            )
            output_url = (
                "https://github.com/PINA-org/PINA-Tutorials/raw/refs/heads/main"
                "/tutorials/tutorial7/data/pinn_solution_0.5_0.5"
            )

            # Define input and output data
            input_data = _load_tensor_from_url(
                input_url, ["x", "y", "mu1", "mu2"]
            )
            output_data = _load_tensor_from_url(output_url, ["u"])

            # Add the "data" condition
            if input_data is not None and output_data is not None:
                n_data = int(input_data.shape[0] * data_size)
                self.conditions["data"] = InputTargetCondition(
                    input=input_data[:n_data], target=output_data[:n_data]
                )
                self.conditions["data"].problem = self
                self.conditions["data"].name = "data"
